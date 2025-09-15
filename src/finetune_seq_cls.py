# -*- coding: utf-8 -*-

from dotenv import load_dotenv
import os

import argparse

import numpy as np
import evaluate
from datasets import load_dataset
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

os.environ["WANDB_PROJECT"] = "billm_test"

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str,
                    help='Specify model_name_or_path to set transformer backbone.')
parser.add_argument('--dataset_name_or_path', type=str, default='dala',
                    help='Specify huggingface dataset name or local file path. Default is dala.')
parser.add_argument('--epochs', type=int, default=10, help='Specify number of epochs, default 10')
parser.add_argument('--batch_size', type=int, default=8, help='Specify number of batch size, default 8')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Specify learning rate, default 1e-4')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Specify weight decay, default 0.01')
parser.add_argument('--max_length', type=int, default=64, help='Specify max length, default 64')
parser.add_argument('--use_peft', type=int, default=1, choices=[0, 1], help='Specify whether to use PEFT (LoRA), default 1 (enabled)')
parser.add_argument('--lora_r', type=int, default=12, help='Specify lora r, default 12')
parser.add_argument('--lora_alpha', type=int, default=32, help='Specify lora alpha, default 32')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='Specify lora dropout, default 0.1')
# configure hub
parser.add_argument('--push_to_hub', type=int, default=0, choices=[0, 1], help='Specify push_to_hub, default 0')
parser.add_argument('--hub_model_id', type=str, default=None,
                    help='Specify push_to_hub_model_id, default None, format like organization/model_id')
args = parser.parse_args()
print(f'Args: {args}')



# Load dataset and inspect structure
if args.dataset_name_or_path == 'dala':
    ds = load_dataset("giannor/dala", token=hf_token)
    label2id = {"correct": 0, "incorrect": 1}
elif args.dataset_name_or_path == 'scala':
    ds = load_dataset("EuroEval/scala-da", token=hf_token_euroeval)
    label2id = {"correct": 0, "incorrect": 1}
elif args.dataset_name_or_path == 'angry_tweets':
    ds = load_dataset("EuroEval/angry-tweets-mini", token=hf_token_euroeval)
    label2id = {"positive": 0, "neutral": 1, "negative": 2}
else:
    # For custom datasets, you'll need to define label2id mapping
    ds = load_dataset(args.dataset_name_or_path)
    # Automatically infer labels from the dataset
    if 'train' in ds:
        unique_labels = set(ds['train']['label'])
        label2id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    else:
        raise ValueError("Please define label2id mapping for your custom dataset")

id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)

print(f"Detected {num_labels} labels: {label2id}")

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path, 
    token=hf_token, 
    num_labels=num_labels,
    # force_download=True
)
if 'mistral' in args.model_name_or_path.lower():
    tokenizer.add_special_tokens({'pad_token': '<unk>'})
elif tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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
    raise NotImplementedError(
        f"Model {args.model_name_or_path} not supported.")


model = MODEL.from_pretrained(
    args.model_name_or_path,
    num_labels=num_labels,
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


# Fix 1: Updated preprocessing function
def preprocess_function(examples):
    """Tokenize the texts and prepare labels"""
    # Handle different dataset formats
    if 'text' in examples:
        texts = examples['text']
    elif 'sentence' in examples:  # For SST2
        texts = examples['sentence']
    else:
        raise ValueError("Dataset should have 'text' or 'sentence' column")

    # Tokenize
    result = tokenizer(
        texts,
        truncation=True,
        padding=False,  # Will be handled by data collator
        max_length=args.max_length
    )

    # Handle labels - Fix for nested labels
    if 'label' in examples:
        labels = examples['label']
        # Convert string labels to integers if needed
        if isinstance(labels[0], str):
            result['labels'] = [label2id[label] for label in labels]
        else:
            result['labels'] = labels

    return result


# Tokenize datasets and remove original columns that might interfere
tokenized_ds = ds.map(preprocess_function, batched=True, remove_columns=ds["train"].column_names)

# Data collator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

# Load metrics
accuracy = evaluate.load("accuracy")
mcc = evaluate.load("matthews_correlation")
f1 = evaluate.load("f1")


# Updated compute_metrics function for multiclass
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    mcc_score = mcc.compute(predictions=predictions, references=labels)

    # Handle F1 for multiclass
    if num_labels > 2:
        f1_score = f1.compute(predictions=predictions, references=labels, average='weighted')
    else:
        f1_score = f1.compute(predictions=predictions, references=labels)

    return {
        "accuracy": acc["accuracy"],
        "matthews_correlation": mcc_score["matthews_correlation"],
        "f1": f1_score["f1"]
    }

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
    push_to_hub=args.push_to_hub,
    hub_model_id=args.hub_model_id,
    report_to="wandb"
)

# # Training arguments BiLLM like
# training_args = TrainingArguments(
#     output_dir=output_dir,
#     learning_rate=args.learning_rate,
#     per_device_train_batch_size=args.batch_size,
#     per_device_eval_batch_size=args.batch_size,
#     num_train_epochs=args.epochs,
#     weight_decay=args.weight_decay,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     report_to="wandb",
#     push_to_hub=args.push_to_hub,
#     hub_model_id=args.hub_model_id,
# )

patience = 20
# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"] if "validation" in tokenized_ds else tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)], # EuroStyle
)

# Train the model
trainer.train()

# Push the best model to the hub
if args.push_to_hub:
    trainer.push_to_hub()

# Evaluate the model
eval_results = trainer.evaluate(eval_dataset=tokenized_ds["test"])
print(f"Eval results: {eval_results}")

"""
Examples:
# PEFT (LoRA) fine-tuning (default)
python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step31816_2dyna
python src/finetune_seq_cls.py --model_name_or_path google/gemma-3-12b-pt --dataset_name_or_path scala
python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path angry_tweets
python src/finetune_seq_cls.py --model_name_or_path google/gemma-3-4b-pt --dataset_name_or_path angry_tweets

# Full fine-tuning (no PEFT)
python src/finetune_seq_cls.py --model_name_or_path google/gemma-3-4b-pt --dataset_name_or_path angry_tweets --use_peft 0
python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path scala --use_peft 0
"""