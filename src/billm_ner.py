# -*- coding: utf-8 -*-

import argparse

import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, EarlyStoppingCallback
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import OptimizerNames, TrainingArguments
from transformers import DataCollatorForTokenClassification
from transformers import Trainer

from peft import get_peft_model, LoraConfig, TaskType
from billm import LlamaForTokenClassification, MistralForTokenClassification, Gemma3ForTokenClassification
from dotenv import load_dotenv
import os
import torch

envs_dir = os.getcwd() + '/envs.env'
print(f"Loading envs from: {envs_dir}")
load_dotenv(envs_dir)
hf_token = os.getenv("HF_TOKEN")
hf_token_euroeval = os.getenv("HF_TOKEN_EUROEVAL")
wandb_api_key = os.getenv("WANDB_API_KEY")

os.environ["WANDB_PROJECT"] = "billm_test"


parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='NousResearch/Llama-2-7b-hf',
                    help='Specify model_name_or_path to set transformer backbone. Default is NousResearch/Llama-2-7b-hf')
parser.add_argument('--dataset_name_or_path', type=str, default='conll2003',
                    help='Specify huggingface dataset name or local file path. Default is conll2003.')
parser.add_argument('--epochs', type=int, default=10, help='Specify number of epochs, default 10')
parser.add_argument('--batch_size', type=int, default=8, help='Specify number of batch size, default 8')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Specify learning rate, default 1e-4')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Specify weight decay, default 0.01')
parser.add_argument('--max_length', type=int, default=64, help='Specify max length, default 64')
parser.add_argument('--use_peft', type=int, default=1, choices=[0, 1], help='Specify whether to use PEFT (LoRA), default 1 (enabled)')
parser.add_argument('--lora_r', type=int, default=32, help='Specify lora r, default 32')
parser.add_argument('--lora_alpha', type=int, default=32, help='Specify lora alpha, default 32')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='Specify lora dropout, default 0.1')
# configure hub
parser.add_argument('--push_to_hub', type=int, default=0, choices=[0, 1], help='Specify push_to_hub, default 0')
parser.add_argument('--hub_model_id', type=str, default=None,
                    help='Specify push_to_hub_model_id, default None, format like organization/model_id')
args = parser.parse_args()
print(f'Args: {args}')


tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=hf_token)
if 'mistral' in args.model_name_or_path.lower():
    tokenizer.add_special_tokens({'pad_token': '<unk>'})
elif tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

seqeval = evaluate.load("seqeval")
if args.dataset_name_or_path == 'wnut_17':
    ds = load_dataset("wnut_17")
    label2id = {"O": 0, "B-corporation": 1, "I-corporation": 2, "B-creative-work": 3, "I-creative-work": 4, "B-group": 5, "I-group": 6, "B-location": 7, "I-location": 8, "B-person": 9, "I-person": 10, "B-product": 11, "I-product": 12, }
elif args.dataset_name_or_path == 'conll2003':
    ds = load_dataset("conll2003")
    label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
elif args.dataset_name_or_path == 'dansk':
    ds = load_dataset("EuroEval/dansk-mini", token=hf_token_euroeval)
    # label2id = {"o": 0, "b-loc": 1, "i-loc": 2, "b-org": 3, "i-org": 4, "b-per": 5, "i-per": 6, "b-misc": 7, "i-misc": 8}
    label2id = {"O": 0, "B-LOC": 1, "I-LOC": 2, "B-ORG": 3, "I-ORG": 4, "B-PER": 5, "I-PER": 6, "B-MISC": 7, "I-MISC": 8}
else:
    raise NotImplementedError

id2label = {v: k for k, v in label2id.items()}
label_list = list(label2id.keys())

# Initialize model based on model name
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
if 'mistral' in args.model_name_or_path.lower():
    MODEL = MistralForTokenClassification
elif 'llama' in args.model_name_or_path.lower():
    MODEL = LlamaForTokenClassification
elif any(x in args.model_name_or_path.lower() for x in ['gemma3', 'gemma-3', 'gemma_3', 'student']):
    MODEL = Gemma3ForTokenClassification
else:
    raise NotImplementedError(f"Model {args.model_name_or_path} not supported.")

model = MODEL.from_pretrained(
    args.model_name_or_path, 
    num_labels=len(label2id), 
    id2label=id2label, 
    label2id=label2id, 
    token=hf_token
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
        task_type=TaskType.TOKEN_CLS,
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

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, padding='longest', max_length=args.max_length, truncation=True)

    if args.dataset_name_or_path == 'dansk':
        label_column = "labels"
    else:
        label_column = "ner_tags"

    labels = []
    for i, label in enumerate(examples[label_column]):
        if args.dataset_name_or_path == 'dansk':
            label = [label2id[l] for l in label]
        # print(label)
        # if i == 10:
        #     exit(1)
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def remove_misc(predictions, labels):
    """Remove B-MISC and I-MISC (case-insensitive) from both predictions and labels while preserving alignment."""
    filtered_predictions = []
    filtered_labels = []
    
    for pred_seq, label_seq in zip(predictions, labels):
        filtered_pred_seq = []
        filtered_label_seq = []
        
        for pred_tag, label_tag in zip(pred_seq, label_seq):
            # Only keep the pair if neither is a MISC tag
            if (pred_tag.lower() not in {"b-misc", "i-misc"} and 
                label_tag.lower() not in {"b-misc", "i-misc"}):
                filtered_pred_seq.append(pred_tag)
                filtered_label_seq.append(label_tag)
        
        filtered_predictions.append(filtered_pred_seq)
        filtered_labels.append(filtered_label_seq)
    
    return filtered_predictions, filtered_labels


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)

    # Micro F1 without MISC tags
    preds_no_misc, labels_no_misc = remove_misc(true_predictions, true_labels)
    results_no_misc = seqeval.compute(predictions=preds_no_misc, references=labels_no_misc)

    return {
        "f1_no_misc": results_no_misc["overall_f1"],
        "f1": results["overall_f1"],
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "accuracy": results["overall_accuracy"],
    }


tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

i = 1
# Include PEFT in output directory name
peft_suffix = "peft" if args.use_peft else "full"
output_dir = f"billm_{args.dataset_name_or_path.replace('/', '-')}_{args.model_name_or_path.replace('/', '-')}_{peft_suffix}_ckpt".replace('.', '').replace('_-', '_').replace('-_', '_')

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
#     push_to_hub=args.push_to_hub,
#     hub_model_id=args.hub_model_id,
#     report_to="wandb",
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

trainer.train()

# Evaluate the model
eval_results = trainer.evaluate(eval_dataset=tokenized_ds["test"])
print(f"Eval results on test set: {eval_results}")

# push the best model to the hub
if args.push_to_hub:
    trainer.push_to_hub()


"""

python src/billm_ner.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path dansk
python src/billm_ner.py --model_name_or_path google/gemma-3-4b-pt --dataset_name_or_path dansk
python src/billm_ner.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path dansk --use_peft 0

"""
