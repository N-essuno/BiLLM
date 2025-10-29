# -*- coding: utf-8 -*-

from dotenv import load_dotenv
import os

import argparse

# Parse arguments first to set GPU device early
parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str,
                    help='Specify model_name_or_path to set transformer backbone.')
parser.add_argument('--dataset_name_or_path', type=str, default='dala',
                    help='Specify huggingface dataset name or local file path. Default is dala.')
parser.add_argument('--epochs', type=int, default=10, help='Specify number of epochs, default 10')
parser.add_argument('--batch_size', type=int, default=128, help='Specify number of batch size, default 128') # 128 instead of 8
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Specify learning rate, default 1e-5') # 1e-5 instead of 1e-4
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Specify gradient accumulation steps, default 1')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Specify weight decay, default 0.0')
parser.add_argument('--max_length', type=int, default=2048, help='Specify max length, default 2048') # 2048 instead of 64
parser.add_argument('--use_peft', type=int, default=1, choices=[0, 1], help='Specify whether to use PEFT (LoRA), default 1 (enabled)')
parser.add_argument('--lora_r', type=int, default=16, help='Specify lora r, default 16') # 16 instead of 12
parser.add_argument('--lora_alpha', type=int, default=32, help='Specify lora alpha, default 32')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='Specify lora dropout, default 0.1')
parser.add_argument('--freeze_base_model', type=int, default=0, choices=[0, 1], help='Specify whether to freeze base model and train only classification head, default 0 (disabled)')
# configure hub
parser.add_argument('--push_to_hub', type=int, default=0, choices=[0, 1], help='Specify push_to_hub, default 0')
parser.add_argument('--hub_model_id', type=str, default=None,
                    help='Specify push_to_hub_model_id, default None, format like organization/model_id')
# configure device
parser.add_argument('--gpu_device', type=str, default=None,
                    help='Specify which GPU device to use (0, 1, 2, etc.) or "all" to use all available GPUs. If not specified, uses default CUDA device or auto-selects.')
parser.add_argument('--run_name_suffix', type=str, default='', help='Specify run name suffix, default empty string')
args = parser.parse_args()

# Set CUDA device as early as possible if specified
if args.gpu_device is not None and args.gpu_device != "all":
    # Convert to int if it's a numeric string
    try:
        gpu_id = int(args.gpu_device)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"Set CUDA_VISIBLE_DEVICES to: {gpu_id}")
    except ValueError:
        raise ValueError(f"Invalid gpu_device value: {args.gpu_device}. Must be an integer or 'all'.")
elif args.gpu_device == "all":
    print("Using all available GPUs for training")
    # Don't set CUDA_VISIBLE_DEVICES when using all GPUs

import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, EarlyStoppingCallback, TrainerCallback
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import OptimizerNames
from peft import get_peft_model, LoraConfig, TaskType
import wandb
from billm import LlamaForSequenceClassification, MistralForSequenceClassification, Qwen2ForSequenceClassification, OpenELMForSequenceClassification, Gemma3ForSequenceClassification
import torch

envs_dir = os.getcwd() + '/envs.env'
print(f"Loading envs from: {envs_dir}")
load_dotenv(envs_dir)
hf_token = os.getenv("HF_TOKEN")
hf_token_euroeval = os.getenv("HF_TOKEN_EUROEVAL")
wandb_api_key = os.getenv("WANDB_API_KEY")

os.environ["WANDB_PROJECT"] = "billm_test"

print(f'Args: {args}')

class BestMetricsLoggerCallback(TrainerCallback):
    def __init__(self):
        # {metric_name: (best_value, full_metrics_dict)}
        self.best_metrics = {}
        self.metric_keywords = [
            "precision", "recall", "matthews_correlation", "mcc", "accuracy", "loss", "f1"
        ]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        for metric, value in metrics.items():
            # Only track metrics containing specified keywords and are float/int
            if (
                isinstance(value, (float, int)) and
                any(keyword in metric.lower() for keyword in self.metric_keywords)
            ):
                # For loss, lower is better; for others, higher is better
                if "loss" in metric.lower():
                    is_better = (metric not in self.best_metrics) or (value < self.best_metrics[metric][0])
                else:
                    is_better = (metric not in self.best_metrics) or (value > self.best_metrics[metric][0])
                if is_better:
                    self.best_metrics[metric] = (value, metrics.copy())

    def on_train_end(self, args, state, control, **kwargs):
        print("\n========== BEST METRICS SUMMARY ==========")
        for metric, (best_value, metrics_dict) in self.best_metrics.items():
            print(f"\nBest {metric}: {best_value:.4f}")
            print("Related metrics for this run:")
            for k, v in metrics_dict.items():
                if (
                    isinstance(v, (float, int)) and
                    any(keyword in k.lower() for keyword in self.metric_keywords)
                ):
                    print(f"  {k}: {v:.4f}")
                elif any(keyword in k.lower() for keyword in self.metric_keywords):
                    print(f"  {k}: {v}")

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
elif any(x in args.model_name_or_path.lower() for x in ['llama', 'munin']):
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
# Clear CUDA cache to avoid memory issues from previous runs
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if torch.backends.mps.is_available() and args.gpu_device is None:
    device = torch.device("mps")
    model = model.to(device)
    print("Using MPS device. bfloat16 is not supported, using default dtype.")
elif torch.cuda.is_available():
    if args.gpu_device == "all":
        # Use all available GPUs
        if torch.cuda.device_count() > 1:
            print(f"Using all {torch.cuda.device_count()} available GPUs with DataParallel")
            device = torch.device("cuda")
            model = model.to(device).bfloat16()
            # DataParallel will be applied by the Trainer automatically when multiple GPUs are detected
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("Only 1 GPU available, using single GPU mode")
            device = torch.device("cuda:0")
            model = model.to(device).bfloat16()
            print(f"Using CUDA device 0: {torch.cuda.get_device_name(0)} with bfloat16.")
    elif args.gpu_device is not None:
        # Use specific GPU device
        try:
            gpu_id = int(args.gpu_device)
            # When CUDA_VISIBLE_DEVICES is set, PyTorch sees only the specified GPU as device 0
            if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                raise ValueError(f"GPU device {gpu_id} not available or CUDA not accessible")
            device = torch.device("cuda:0")  # Always use device 0 since CUDA_VISIBLE_DEVICES isolates the GPU
            # Ensure we're using the correct device
            with torch.cuda.device(0):
                model = model.to(device).bfloat16()
            print(f"Using CUDA device {gpu_id} (mapped to cuda:0): {torch.cuda.get_device_name(0)} with bfloat16.")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
        except ValueError:
            raise ValueError(f"Invalid gpu_device value: {args.gpu_device}. Must be an integer or 'all'.")
    else:
        device = torch.device("cuda")
        model = model.to(device).bfloat16()
        print(f"Using default CUDA device: {torch.cuda.get_device_name()} with bfloat16.")
else:
    if args.gpu_device is not None:
        print(f"Warning: GPU device {args.gpu_device} specified but CUDA not available. Using CPU.")
    device = torch.device("cpu")
    model = model.to(device)
    print("Using CPU device. bfloat16 is not used.")

# Configure training mode
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
elif args.freeze_base_model:
    print("Freezing base model, training only classification head...")
    # Freeze all parameters in the base model
    for param in model.model.parameters():
        param.requires_grad = False
    
    # Keep classification head trainable
    for param in model.score.parameters():
        param.requires_grad = True
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} || All params: {total_params:,} || Trainable%: {100 * trainable_params / total_params:.2f}")
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

# Analyze token distribution and memory requirements
def analyze_token_distribution(dataset, dataset_name="train"):
    """Analyze token count distribution in the dataset"""
    token_counts = [len(example['input_ids']) for example in dataset]
    
    print(f"\n{'='*60}")
    print(f"TOKEN ANALYSIS FOR {dataset_name.upper()} SET")
    print(f"{'='*60}")
    print(f"Total examples: {len(token_counts):,}")
    print(f"Min tokens per example: {min(token_counts)}")
    print(f"Max tokens per example: {max(token_counts)}")
    print(f"Average tokens per example: {np.mean(token_counts):.1f}")
    print(f"Median tokens per example: {np.median(token_counts):.1f}")
    print(f"Token count standard deviation: {np.std(token_counts):.1f}")
    
    # Percentile analysis
    percentiles = [50, 75, 90, 95, 99]
    print(f"\nToken count percentiles:")
    for p in percentiles:
        print(f"  {p}th percentile: {np.percentile(token_counts, p):.0f} tokens")
    
    # Memory estimation
    batch_size = args.batch_size
    print(f"\n{'='*60}")
    print(f"MEMORY ESTIMATION (per device)")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size}")
    print(f"Max sequence length: {args.max_length}")
    
    # Calculate tokens per batch scenarios
    avg_tokens_per_batch = batch_size * np.mean(token_counts)
    max_tokens_per_batch = batch_size * args.max_length  # Worst case with padding
    
    print(f"\nTokens per batch scenarios:")
    print(f"  Average case: {avg_tokens_per_batch:,.0f} tokens per batch")
    print(f"  Worst case (max padding): {max_tokens_per_batch:,.0f} tokens per batch")
    
    # Check if gradient accumulation is used
    if args.gradient_accumulation_steps is not None:
        grad_accum_steps = args.gradient_accumulation_steps
        print(f"  Gradient accumulation steps: {grad_accum_steps}")
        print(f"  Effective batch size: {batch_size * grad_accum_steps}")
        print(f"  Total tokens per gradient update (avg): {avg_tokens_per_batch * grad_accum_steps:,.0f}")
        print(f"  Total tokens per gradient update (max): {max_tokens_per_batch * grad_accum_steps:,.0f}")
    
    # Model parameter estimation (rough)
    total_params = sum(p.numel() for p in model.parameters())
    if args.use_peft:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel parameters (LoRA):")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    else:
        print(f"\nModel parameters (Full fine-tuning):")
        print(f"  Total trainable parameters: {total_params:,}")
    
    print(f"\n{'='*60}")
    return token_counts

# Analyze train and validation sets
# print("Analyzing token distribution...")
# train_token_counts = analyze_token_distribution(tokenized_ds["train"], "train")
# if "validation" in tokenized_ds:
#     val_token_counts = analyze_token_distribution(tokenized_ds["validation"], "validation")
# elif "test" in tokenized_ds:
#     test_token_counts = analyze_token_distribution(tokenized_ds["test"], "test")

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

model_name = args.model_name_or_path.split("/")[-1]

actual_batch_size = args.batch_size * args.gradient_accumulation_steps

i = 1
# Include training type in output directory name
if args.use_peft:
    training_suffix = "lora"
elif args.freeze_base_model:
    training_suffix = "head_only"
else:
    training_suffix = "full"

run_name_suffix = args.run_name_suffix
output_dir = f"{args.dataset_name_or_path.replace('/', '-')}_{model_name}_{training_suffix}_{actual_batch_size}_{i}_{run_name_suffix}".replace('.', '').replace('_-', '_').replace('-_', '_')

# Check if output_dir exists, if so, increment i
while os.path.exists(output_dir):
    i += 1
    output_dir = f"{args.dataset_name_or_path.replace('/', '-')}_{model_name}_{training_suffix}_{actual_batch_size}_{i}_{run_name_suffix}".replace('.', '').replace('_-', '_').replace('-_', '_')

print(f"Output directory: {output_dir}")

wandb.init(name=output_dir)

# EuroStyle - Adapted from src/euroeval/finetuning.py
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy=IntervalStrategy.EPOCH,
    save_strategy=IntervalStrategy.EPOCH,
    logging_steps=30,
    num_train_epochs=args.epochs,
    save_total_limit=1,
    per_device_train_batch_size=args.batch_size,  # Default varies
    per_device_eval_batch_size=args.batch_size,
    eval_accumulation_steps=32,
    optim=OptimizerNames.ADAMW_TORCH,
    learning_rate=args.learning_rate,  # EuroEval default is 2e-5
    warmup_ratio=0.01,   # 1% warmup
    gradient_accumulation_steps= args.gradient_accumulation_steps,
    load_best_model_at_end=True,
    push_to_hub=args.push_to_hub,
    hub_model_id=args.hub_model_id,
    report_to="wandb",
    max_grad_norm=1.0,
    weight_decay=args.weight_decay,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"] if "validation" in tokenized_ds else tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[BestMetricsLoggerCallback()],
)

# Train the model
trainer.train()

# Push the best model to the hub
if args.push_to_hub:
    trainer.push_to_hub()

# Evaluate the model
eval_results = trainer.evaluate(eval_dataset=tokenized_ds["test"], metric_key_prefix="test")


# Print training arguments and evaluation results
print(f"Training arguments: {training_args}")
print("=" * 60)
print(f"Test results: {eval_results}")





"""
Examples:
# PEFT (LoRA) fine-tuning (default)
python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step31816_2dyna
python src/finetune_seq_cls.py --model_name_or_path google/gemma-3-12b-pt --dataset_name_or_path scala
python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path angry_tweets
python src/finetune_seq_cls.py --model_name_or_path google/gemma-3-4b-pt --dataset_name_or_path angry_tweets
python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path angry_tweets

# Full fine-tuning (no PEFT)
python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path angry_tweets --use_peft 0

Latest runs

python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path angry_tweets --use_peft 0

python src/finetune_seq_cls.py --model_name_or_path google/gemma-3-1b-pt --dataset_name_or_path angry_tweets --use_peft 0

python src/finetune_seq_cls.py --model_name_or_path google/gemma-3-4b-pt --dataset_name_or_path angry_tweets --use_peft 0 --gpu_device 0 --batch_size 8 --gradient_accumulation_steps 16

python src/finetune_seq_cls.py --model_name_or_path google/gemma-3-4b-it --dataset_name_or_path angry_tweets --use_peft 0 --gpu_device 1 --batch_size 8 --gradient_accumulation_steps 16

python src/finetune_seq_cls.py --model_name_or_path google/gemma-3-1b-it --dataset_name_or_path angry_tweets --use_peft 0 --gpu_device 0

python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step3678_onpolicy --dataset_name_or_path angry_tweets --use_peft 0 --gpu_device 0

python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step7356_cos_giga_distill_full --dataset_name_or_path angry_tweets --use_peft 0 --gpu_device 0

python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step15908_dyna_none --dataset_name_or_path angry_tweets --use_peft 0 --gpu_device 0

python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step24125_dyna_commonpile --dataset_name_or_path angry_tweets --use_peft 0 --gpu_device 0

python src/finetune_seq_cls.py --model_name_or_path google/gemma-3-1b-pt --dataset_name_or_path scala --use_peft 0 --gpu_device 0

python src/finetune_seq_cls.py --model_name_or_path google/gemma-3-1b-it --dataset_name_or_path scala --use_peft 0 --gpu_device 0

python src/finetune_seq_cls.py --model_name_or_path google/gemma-3-4b-pt --dataset_name_or_path scala --use_peft 0 --batch_size 8 --gradient_accumulation_steps 16 --gpu_device 0

python src/finetune_seq_cls.py --model_name_or_path google/gemma-3-4b-it --dataset_name_or_path scala --use_peft 0 --batch_size 8 --gradient_accumulation_steps 16 --gpu_device 0

python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path scala --use_peft 0 --gpu_device 0


to run



python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step15908_dyna_none --dataset_name_or_path scala --use_peft 0 --gpu_device 0




tests

python src/finetune_seq_cls.py --model_name_or_path ../../production/models/munin-7b-core-pt-3 --dataset_name_or_path angry_tweets --use_peft 1 --gpu_device 0 --batch_size 128 --lora_r 8 --lora_alpha 16 --batch_size 8 --gradient_accumulation_steps 16 --learning_rate 2e-5 --run_name_suffix lukas

python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path angry_tweets --use_peft 1 --gpu_device 0 --batch_size 128 --lora_r 8 --lora_alpha 16 --learning_rate 2e-5 --run_name_suffix lukas


python src/finetune_seq_cls.py --model_name_or_path google/gemma-3-1b-pt --dataset_name_or_path angry_tweets --use_peft 1 --gpu_device 0 --batch_size 128 --lora_r 4 --lora_alpha 8 --run_name_suffix lukas

python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step7356_cos_giga_distill_full --dataset_name_or_path angry_tweets --use_peft 1 --gpu_device 0 --batch_size 128 --lora_r 8 --lora_alpha 16 --run_name_suffix lukas

python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step24125_dyna_commonpile --dataset_name_or_path angry_tweets --use_peft 1 --gpu_device 0 --batch_size 128 --lora_r 8 --lora_alpha 16 --run_name_suffix lukas

python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path angry_tweets --use_peft 1 --gpu_device 0 --batch_size 128 --lora_r 8 --lora_alpha 16 --run_name_suffix lukas

python src/finetune_seq_cls.py --model_name_or_path google/gemma-3-1b-pt --dataset_name_or_path angry_tweets --use_peft 1 --gpu_device 0 --batch_size 128 --lora_r 8 --lora_alpha 16 --run_name_suffix lukas

python src/finetune_seq_cls.py --model_name_or_path google/gemma-3-12b-pt --dataset_name_or_path angry_tweets --use_peft 1 --gpu_device 0 --batch_size 8 --gradient_accumulation_steps 16 --lora_r 8 --lora_alpha 16 --run_name_suffix lukas

python src/finetune_seq_cls.py --model_name_or_path google/gemma-3-4b-pt --dataset_name_or_path angry_tweets --use_peft 0 --gpu_device 0 --batch_size 8 --gradient_accumulation_steps 16  --run_name_suffix lukas

python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path angry_tweets --use_peft 0 --gpu_device 0 --learning_rate 1e-6 --run_name_suffix lukas

python src/finetune_seq_cls.py --model_name_or_path google/gemma-3-4b-pt --dataset_name_or_path angry_tweets --use_peft 0 --gpu_device 0 --batch_size 8 --gradient_accumulation_steps 16 --freeze_base_model 1 --run_name_suffix lukas

python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path angry_tweets --use_peft 0 --batch_size 8 --gradient_accumulation_steps 32 --gpu_device 0

python src/finetune_seq_cls.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path angry_tweets --use_peft 0 --gpu_device 0 --run_name_suffix lukas

python src/finetune_seq_cls.py --model_name_or_path google/gemma-3-1b-pt --dataset_name_or_path angry_tweets --use_peft 0 --gpu_device 0 --weight_decay 0.01 --run_name_suffix lukas

"""