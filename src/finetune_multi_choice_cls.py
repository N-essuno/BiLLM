# -*- coding: utf-8 -*-

from dotenv import load_dotenv
import os

import argparse
import hashlib
import re
import numpy as np
import evaluate
from datasets import load_dataset, DatasetDict
from transformers import AutoConfig, AutoTokenizer, EarlyStoppingCallback
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from billm import LlamaForSequenceClassification, MistralForSequenceClassification, Qwen2ForSequenceClassification, OpenELMForSequenceClassification, Gemma3ForSequenceClassification
import torch

from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import OptimizerNames, TrainingArguments

envs_dir = os.getcwd() + '/envs.env'
print(f"Loading envs from: {envs_dir}")
load_dotenv(envs_dir)
hf_token = os.getenv("HF_TOKEN")
hf_token_euroeval = os.getenv("HF_TOKEN_EUROEVAL")
wandb_api_key = os.getenv("WANDB_API_KEY")

print(hf_token_euroeval)

os.environ["WANDB_PROJECT"] = "billm_test"

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str,
                    help='Specify model_name_or_path to set transformer backbone.')
parser.add_argument('--dataset_name_or_path', type=str, required=True,
                    help='Specify huggingface dataset name or local file path.')
parser.add_argument('--epochs', type=int, default=10, help='Specify number of epochs, default 10')
parser.add_argument('--batch_size', type=int, default=8, help='Specify batch size, default 8')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Specify learning rate, default 1e-4')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Specify weight decay, default 0.01')
parser.add_argument('--max_length', type=int, default=64, help='Specify max length, default 64')
parser.add_argument('--use_peft', type=int, default=1, choices=[0, 1], help='Specify whether to use PEFT (LoRA), default 1 (enabled)')
parser.add_argument('--lora_r', type=int, default=12, help='Specify lora r, default 12')
parser.add_argument('--lora_alpha', type=int, default=32, help='Specify lora alpha, default 32')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='Specify lora dropout, default 0.1')
parser.add_argument('--push_to_hub', type=int, default=0, choices=[0, 1], help='Specify push_to_hub, default 0')
parser.add_argument('--hub_model_id', type=str, default=None, help='Hub model ID for pushing')

args = parser.parse_args()
print(f'Args: {args}')

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path, 
    token=hf_token, 
    # force_download=True
)
if 'mistral' in args.model_name_or_path.lower():
    tokenizer.add_special_tokens({'pad_token': '<unk>'})
elif tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Load metrics
accuracy_metric = evaluate.load("accuracy")
matthews_metric = evaluate.load("matthews_correlation")

def prepare_multiple_choice_examples(examples):
    """
    Prepare multiple choice examples following EuroEval's approach.
    Converts each question into multiple binary classification examples.
    """
    new_examples = {
        "input_ids": [],
        "attention_mask": [],
        "token_type_ids": [],
        "label": [],
        "id": []
    }
    
    for doc in examples["text"]:
        sections = doc.split("\n")
        
        # Find choice options (lines starting with letter + dot)
        candidate_choice_idxs = [
            idx for idx, section in enumerate(sections)
            if re.match(pattern=r"^[a-z]\. ", string=section) is not None
        ]
        
        # Get the final contiguous block of choices
        choice_idxs = []
        for idx in reversed(candidate_choice_idxs):
            if len(choice_idxs) < 2 or (len(choice_idxs) >= 2 and idx == choice_idxs[-1] - 1):
                choice_idxs.append(idx)
        
        choices = [sections[idx] for idx in reversed(choice_idxs)]
        
        # Extract context and question (everything before choices)
        question_idx = min(choice_idxs) - 2
        context_and_question = "\n".join(sections[:question_idx + 1]).strip()
        
        # Tokenize context+question paired with each choice
        tokenized = tokenizer(
            text=[context_and_question] * len(choices),
            text_pair=[choice[3:] for choice in choices],  # Remove "x. " prefix
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors=None
        )
        
        # Create binary labels (1 for correct choice, 0 for others)
        correct_letter = examples["label"][0] if "label" in examples else "a"
        labels = [
            int(choice.startswith(f"{correct_letter}. "))
            for choice in choices
        ]
        
        # Add to new examples
        for i in range(len(choices)):
            new_examples["input_ids"].append(tokenized["input_ids"][i])
            new_examples["attention_mask"].append(tokenized["attention_mask"][i])
            if "token_type_ids" in tokenized:
                new_examples["token_type_ids"].append(tokenized["token_type_ids"][i])
            new_examples["label"].append(labels[i])
            new_examples["id"].append(hashlib.md5(doc.encode()).hexdigest())
    
    return new_examples

# Load and prepare dataset
if args.dataset_name_or_path == "danske_talemaader":
    dataset = load_dataset("EuroEval/danske-talemaader", token=hf_token_euroeval)
else:
    dataset = load_dataset(args.dataset_name_or_path)


processed_dataset = dataset.map(
    prepare_multiple_choice_examples,
    batched=True,
    batch_size=1,  # Process one question at a time
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
)

# Initialize model based on model name
lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
if 'mistral' in args.model_name_or_path.lower():
    MODEL = MistralForSequenceClassification
elif 'llama' in args.model_name_or_path.lower():
    MODEL = LlamaForSequenceClassification
elif 'qwen2' in args.model_name_or_path.lower():
    MODEL = Qwen2ForSequenceClassification
elif 'openelm' in args.model_name_or_path.lower():
    MODEL = OpenELMForSequenceClassification
elif any(x in args.model_name_or_path.lower() for x in ['gemma3', 'gemma-3', 'gemma_3', 'student']):
    MODEL = Gemma3ForSequenceClassification
else:
    raise NotImplementedError(f"Model type not supported: {args.model_name_or_path}")

# Binary classification (correct vs incorrect choice)
id2label = {0: "incorrect", 1: "correct"}
label2id = {v: k for k, v in id2label.items()}

model = MODEL.from_pretrained(
    args.model_name_or_path,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    token=hf_token,
    # force_download=True,
)

# Device and dtype handling
if torch.backends.mps.is_available():
    device = torch.device("mps")
    model = model.to(device)
    print("Using MPS device. bfloat16 is not supported, using default dtype.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    model = model.to(device).bfloat16()
    print("Using CUDA device with bfloat16.")
else:
    device = torch.device("cpu")
    model = model.to(device)
    print("Using CPU device. bfloat16 is not used.")

# Configure LoRA/PEFT if enabled
if args.use_peft:
    print("Applying PEFT (LoRA) configuration...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
else:
    print("Using full fine-tuning (no PEFT)...")
    # For full fine-tuning, all parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} || All params: {total_params:,} || Trainable%: {100 * trainable_params / total_params:.2f}")

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")


def compute_metrics(eval_pred):
    """Compute Matthews correlation and accuracy for binary classification of choices."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    matthews = matthews_metric.compute(predictions=predictions, references=labels)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    
    return {"matthews_correlation": matthews["matthews_correlation"], "accuracy": accuracy["accuracy"]}

def postprocess_predictions(predictions, dataset):
    """
    Convert binary predictions back to multiple choice predictions.
    Group by question ID and use highest confidence correct prediction.
    """
    predictions = np.argmax(predictions, axis=1)
    
    # Group predictions by question ID
    question_groups = {}
    for i, (pred, example_id) in enumerate(zip(predictions, dataset["id"])):
        if example_id not in question_groups:
            question_groups[example_id] = []
        question_groups[example_id].append((i, pred))
    
    # For each question, select the choice with highest "correct" confidence
    final_predictions = []
    final_labels = []
    
    for example_id, choices in question_groups.items():
        # Find indices for this question
        indices = [idx for idx, _ in choices]
        question_preds = [predictions[idx] for idx in indices]
        question_labels = [dataset["label"][idx] for idx in indices]
        
        # Select the choice predicted as correct (if any), otherwise first choice
        correct_choice_idx = 0
        for i, pred in enumerate(question_preds):
            if pred == 1:  # Predicted as correct
                correct_choice_idx = i
                break
        
        # The actual correct choice index
        actual_correct_idx = question_labels.index(1) if 1 in question_labels else 0
        
        final_predictions.append(correct_choice_idx)
        final_labels.append(actual_correct_idx)
    
    return {"predictions": final_predictions, "label_ids": final_labels}

i = 1
# Include PEFT in output directory name
peft_suffix = "peft" if args.use_peft else "full"
output_dir = f"billm_{args.dataset_name_or_path.replace('/', '-')}_{args.model_name_or_path.replace('/', '-')}_{peft_suffix}_{i}".replace('.', '').replace('_-', '_').replace('-_', '_')

# Check if output_dir exists, if so, increment i
while os.path.exists(output_dir):
    i += 1
    output_dir = f"billm_{args.dataset_name_or_path.replace('/', '-')}_{args.model_name_or_path.replace('/', '-')}_{peft_suffix}_ckpt_{i}".replace('.', '').replace('_-', '_').replace('-_', '_')
print(f"Output directory: {output_dir}")

# EuroStyle - Adapted from src/euroeval/finetuning.py
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy=IntervalStrategy.STEPS,
    save_strategy=IntervalStrategy.STEPS,
    eval_steps=30,
    logging_steps=30,
    save_steps=30,
    max_steps=10_000,  # (1 if testing)
    save_total_limit=1,
    per_device_train_batch_size=args.batch_size,  # Default varies
    per_device_eval_batch_size=args.batch_size,
    eval_accumulation_steps=32,
    optim=OptimizerNames.ADAMW_TORCH,
    learning_rate=args.learning_rate,  # EuroEval default is 2e-5
    warmup_ratio=0.01,   # 1% warmup
    gradient_accumulation_steps=32 // args.batch_size,
    load_best_model_at_end=True,
    push_to_hub=bool(args.push_to_hub),
    hub_model_id=args.hub_model_id,
    report_to="wandb"
)

patience = 20
# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"] if "validation" in processed_dataset else processed_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)], # EuroStyle
)

# Train the model
trainer.train()

# Push to hub if requested
if args.push_to_hub:
    trainer.push_to_hub()

# Evaluate the model
eval_results = trainer.evaluate(eval_dataset=processed_dataset["test"])
print(f"Eval results: {eval_results}")

print("Training completed!")

"""
Examples:
# PEFT (LoRA) fine-tuning (default)
python src/finetune_multi_choice_cls.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path danske_talemaader

python src/finetune_multi_choice_cls.py --model_name_or_path google/gemma-3-4b-pt --dataset_name_or_path danske_talemaader

# Full fine-tuning (no PEFT)
python src/finetune_multi_choice_cls.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path danske_talemaader --use_peft 0
"""