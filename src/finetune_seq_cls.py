# -*- coding: utf-8 -*-

from dotenv import load_dotenv
import os

load_dotenv(os.getcwd() + '/../envs.env')
hf_token = os.getenv("HF_TOKEN")

import argparse

import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, EarlyStoppingCallback
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from billm import LlamaForSequenceClassification, MistralForSequenceClassification, Qwen2ForSequenceClassification, \
    OpenELMForSequenceClassification, Gemma3ForSequenceClassification
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str,
                    default='danish-foundation-models/gemma-3-1b-cpt-dynaword-full-v1',
                    help='Specify model_name_or_path to set transformer backbone.')
parser.add_argument('--dataset_name_or_path', type=str, default='dala',
                    help='Specify huggingface dataset name or local file path. Default is dala.')
parser.add_argument('--epochs', type=int, default=3, help='Specify number of epochs, default 3')
parser.add_argument('--batch_size', type=int, default=8, help='Specify number of batch size, default 8')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Specify learning rate, default 1e-4')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Specify weight decay, default 0.01')
parser.add_argument('--max_length', type=int, default=512, help='Specify max length, default 512')
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

# Load dataset and inspect structure (Fix 2: Dataset inspection)
if args.dataset_name_or_path == 'dala':
    ds = load_dataset("giannor/dala", token=hf_token)
    label2id = {"correct": 0, "incorrect": 1}
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

# Initialize model based on model name
if 'mistral' in args.model_name_or_path.lower():
    MODEL = MistralForSequenceClassification
    lora_target_modules = ["q_proj", "v_proj"]
elif 'llama' in args.model_name_or_path.lower():
    MODEL = LlamaForSequenceClassification
    lora_target_modules = ["q_proj", "v_proj"]
elif 'qwen2' in args.model_name_or_path.lower():
    MODEL = Qwen2ForSequenceClassification
    lora_target_modules = ["q_proj", "v_proj"]
elif 'openelm' in args.model_name_or_path.lower():
    MODEL = OpenELMForSequenceClassification
    lora_target_modules = ["q_proj", "v_proj"]
elif any(x in args.model_name_or_path.lower() for x in ['gemma3', 'gemma-3', 'gemma_3']):
    MODEL = Gemma3ForSequenceClassification
    lora_target_modules = ["q_proj", "v_proj"]
else:
    raise NotImplementedError(
        f"Model {args.model_name_or_path} not supported.")

model = MODEL.from_pretrained(
    args.model_name_or_path,
    num_labels=num_labels,
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

# Configure LoRA
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
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load metrics
accuracy = evaluate.load("accuracy")
mcc = evaluate.load("matthews_correlation")
f1 = evaluate.load("f1")


# Fix 3: Updated compute_metrics function for multiclass
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

# training_args = TrainingArguments(
#     output_dir=f"billm_{args.dataset_name_or_path.replace('/', '-')}_{args.model_name_or_path.replace('/', '-')}_ckpt",
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=16,
#     gradient_accumulation_steps=8,
#     learning_rate=2e-4,
#     warmup_steps=30,
#     max_steps=300,
#     lr_scheduler_type="cosine",
#     weight_decay=0.05,
#     max_grad_norm=1.0,
#     gradient_checkpointing=True,
#     metric_for_best_model="matthews_correlation",
#     greater_is_better=True,
#     eval_strategy="steps",
#     eval_steps=15,
#     save_strategy="steps",
#     save_steps=30,
#     load_best_model_at_end=True,
#     push_to_hub=args.push_to_hub,
#     hub_model_id=args.hub_model_id,
# )


# Training arguments
training_args = TrainingArguments(
    output_dir=f"billm_{args.dataset_name_or_path.replace('/', '-')}_{args.model_name_or_path.replace('/', '-')}_ckpt",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="matthews_correlation",
    greater_is_better=True,
    push_to_hub=args.push_to_hub,
    hub_model_id=args.hub_model_id,
)

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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
)

# Train the model
trainer.train()

# Push the best model to the hub
if args.push_to_hub:
    trainer.push_to_hub()