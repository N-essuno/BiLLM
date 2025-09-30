# -*- coding: utf-8 -*-

from dotenv import load_dotenv
import os

import argparse
import hashlib
import re
import numpy as np
import evaluate
from datasets import load_dataset, DatasetDict
from transformers import AutoConfig, AutoTokenizer, EarlyStoppingCallback, TrainerCallback
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from billm import LlamaForSequenceClassification, MistralForSequenceClassification, Qwen2ForSequenceClassification, OpenELMForSequenceClassification, Gemma3ForSequenceClassification
import torch
import wandb
from collections import defaultdict

from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import OptimizerNames, TrainingArguments

USE_CUSTOM_TRAINER = True

envs_dir = os.getcwd() + '/envs.env'
print(f"Loading envs from: {envs_dir}")
load_dotenv(envs_dir)

# # Set memory optimization flags
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

hf_token = os.getenv("HF_TOKEN")
hf_token_euroeval = os.getenv("HF_TOKEN_EUROEVAL")
wandb_api_key = os.getenv("WANDB_API_KEY")

os.environ["WANDB_PROJECT"] = "billm_test"

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str,
                    help='Specify model_name_or_path to set transformer backbone.')
parser.add_argument('--dataset_name_or_path', type=str, required=True,
                    help='Specify huggingface dataset name or local file path.')
parser.add_argument('--epochs', type=int, default=10, help='Specify number of epochs, default 10')
parser.add_argument('--max_steps', type=int, default=10_000, help='Specify max steps, default 10_000')
parser.add_argument('--batch_size', type=int, default=128, help='Specify batch size, default 128') # 128 instead of 8
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Specify learning rate, default 1e-5') # 1e-5 instead of 1e-4
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Specify gradient accumulation steps, default 1')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Specify weight decay, default 0.01')
parser.add_argument('--max_length', type=int, default=2048, help='Specify max length, default 2048') # 2048 instead of 512
parser.add_argument('--use_peft', type=int, default=1, choices=[0, 1], help='Specify whether to use PEFT (LoRA), default 1 (enabled)')
parser.add_argument('--lora_r', type=int, default=16, help='Specify lora r, default 16') # 16 instead of 12
parser.add_argument('--lora_alpha', type=int, default=32, help='Specify lora alpha, default 32')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='Specify lora dropout, default 0.1')
parser.add_argument('--push_to_hub', type=int, default=0, choices=[0, 1], help='Specify push_to_hub, default 0')
parser.add_argument('--hub_model_id', type=str, default=None, help='Hub model ID for pushing')
# configure device
parser.add_argument('--gpu_device', type=int, default=None,
                    help='Specify which GPU device to use (0, 1, 2, etc.). If not specified, uses default CUDA device or auto-selects.')

args = parser.parse_args()

# Set CUDA device as early as possible if specified
if args.gpu_device is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device)
    print(f"Set CUDA_VISIBLE_DEVICES to: {args.gpu_device}")

print(f'Args: {args}')

class BestMetricsLoggerCallback(TrainerCallback):
    def __init__(self):
        # {metric_name: (best_value, full_metrics_dict)}
        self.best_metrics = {}
        self.metric_keywords = [
            "precision", "recall", "matthews_correlation", "mcc", "accuracy", "loss", "multi_choice_accuracy", "multi_choice_matthews_correlation", "f1"
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
        "label": [],
        "id": []
    }
    
    for i, doc in enumerate(examples["text"]):
        sections = doc.split("\n")
        
        # Find choice options (lines starting with letter + dot) - EuroEval uses [a-e]
        choice_idxs = [
            idx
            for idx, section in enumerate(sections)
            if re.match(pattern=r"^[a-e]\. ", string=section) is not None
        ]
        choices = [sections[idx] for idx in choice_idxs]
        
        # Check that the choices are present, and that all of them are at the end
        assert len(choices) > 0, "No choices found in the document."
        assert all(
            choice_idx == len(sections) - i
            for i, choice_idx in enumerate(sorted(choice_idxs, reverse=True), start=1)
        ), "Choices are not at the end of the document."
        
        question_idx = min(choice_idxs) - 2  # -2 to remove the 'Choices:' line
        context_and_question = "\n".join(sections[: question_idx + 1]).strip()
        
        # Tokenize context+question paired with each choice - following EuroEval exactly
        tokenized = tokenizer(
            text=[context_and_question] * len(choices),
            text_pair=[choice[3:] for choice in choices],  # Remove "x. " prefix
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        
        # Create binary labels (1 for correct choice, 0 for others) - EuroEval approach
        correct_letter = examples["label"][i] if "label" in examples else "a"
        labels = [
            int(choice.startswith(f"{letter}. ") and letter == correct_letter)
            for letter, choice in zip("abcde", choices)
        ]
        
        # Print textual conversion example for the first few questions
        if i < 0:  # Optional print
            print(f"\n=== Question {i+1} Conversion Example ===")
            print(f"Original multi-choice question:")
            print(f"Context: {context_and_question[:200]}...")
            print(f"Correct answer: {correct_letter}")
            print(f"\nConverted to {len(choices)} binary classification examples:")
            for j, (choice, label) in enumerate(zip(choices, labels)):
                choice_text = choice[3:]  # Remove "x. " prefix
                print(f"  Binary example {j+1}: Context + '{choice_text}' → Label: {label} ({'correct' if label == 1 else 'incorrect'})")
        
        # Add to new examples - only add fields that exist in tokenized output
        for j in range(len(choices)):
            new_examples["input_ids"].append(tokenized["input_ids"][j])
            new_examples["attention_mask"].append(tokenized["attention_mask"][j])
            new_examples["label"].append(labels[j])
            new_examples["id"].append(hashlib.md5(string=doc.encode()).hexdigest())
    
    return new_examples

# Load and prepare dataset
if args.dataset_name_or_path == "danske_talemaader":
    dataset = load_dataset("EuroEval/danske-talemaader", token=hf_token_euroeval)
elif args.dataset_name_or_path == "danish_citizen_test":
    dataset = load_dataset("EuroEval/danish-citizen-tests-updated", token=hf_token_euroeval)
elif args.dataset_name_or_path == "hellaswag_da":
    dataset = load_dataset("EuroEval/hellaswag-da-mini", token=hf_token_euroeval)
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
# Clear CUDA cache to avoid memory issues from previous runs
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if torch.backends.mps.is_available() and args.gpu_device is None:
    device = torch.device("mps")
    model = model.to(device)
    print("Using MPS device. bfloat16 is not supported, using default dtype.")
elif torch.cuda.is_available():
    if args.gpu_device is not None:
        # When CUDA_VISIBLE_DEVICES is set, PyTorch sees only the specified GPU as device 0
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise ValueError(f"GPU device {args.gpu_device} not available or CUDA not accessible")
        device = torch.device("cuda:0")  # Always use device 0 since CUDA_VISIBLE_DEVICES isolates the GPU
        # Ensure we're using the correct device
        with torch.cuda.device(0):
            model = model.to(device).bfloat16()
        print(f"Using CUDA device {args.gpu_device} (mapped to cuda:0): {torch.cuda.get_device_name(0)} with bfloat16.")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
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


class MultipleChoiceTrainer(Trainer):
    """Custom trainer that computes both binary and multi-choice metrics."""
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to include multi-choice metrics."""
        # Get standard evaluation results
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Only compute multi-choice metrics on test set
        if metric_key_prefix == "test":
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=False,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            
            predictions = output.predictions
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            
            # Convert to multi-choice predictions
            mc_results = postprocess_predictions(predictions, eval_dataset)
            mc_predictions, mc_labels = mc_results["predictions"], mc_results["label_ids"]
            
            # Compute multi-choice accuracy
            mc_accuracy = sum(p == l for p, l in zip(mc_predictions, mc_labels)) / len(mc_predictions)
            eval_results[f"{metric_key_prefix}_multi_choice_accuracy"] = mc_accuracy
            
            # Compute multi-choice Matthews correlation
            mc_matthews = matthews_metric.compute(predictions=mc_predictions, references=mc_labels)
            eval_results[f"{metric_key_prefix}_multi_choice_matthews_correlation"] = mc_matthews["matthews_correlation"]
            
            print(f"\n=== Multi-Choice Evaluation Results ===")
            print(f"Binary classification mcc: {eval_results.get(f'{metric_key_prefix}_matthews_correlation', 'N/A'):.4f}")
            print(f"Binary classification accuracy: {eval_results.get(f'{metric_key_prefix}_accuracy', 'N/A'):.4f}")
            print(f"Multi-choice Matthews correlation: {mc_matthews['matthews_correlation']:.4f}")
            print(f"Multi-choice accuracy: {mc_accuracy:.4f}")
            print(f"Total questions: {len(mc_predictions)}")
            print("========================================\n")
        
        return eval_results

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
    # Use raw predictions (probabilities) instead of argmax for better selection
    if predictions.ndim == 2 and predictions.shape[1] == 2:
        # Use the "correct" class probabilities
        correct_probs = predictions[:, 1]
    else:
        # Fallback to argmax if predictions are already processed
        correct_probs = np.argmax(predictions, axis=1) if predictions.ndim == 2 else predictions
    
    # Group predictions by question ID
    question_groups = defaultdict(list)
    for i, (prob, example_id) in enumerate(zip(correct_probs, dataset["id"])):
        question_groups[example_id].append((i, prob, dataset["label"][i]))
    
    final_predictions = []
    final_labels = []
    
    print(f"\n=== Postprocessing {len(question_groups)} questions ===")
    
    for q_idx, (example_id, choices) in enumerate(question_groups.items()):
        # Extract indices, probabilities, and labels
        indices, probs, labels = zip(*choices)
        
        # Find the choice with highest "correct" probability
        best_choice_idx = np.argmax(probs)
        
        # Find the actual correct choice
        actual_correct_idx = labels.index(1) if 1 in labels else 0
        
        final_predictions.append(best_choice_idx)
        final_labels.append(actual_correct_idx)
        
        # Print first few examples
        if q_idx < 3:
            print(f"Question {q_idx+1}: Predicted choice {best_choice_idx}, Actual choice {actual_correct_idx}")
            print(f"  Choice probabilities: {[f'{p:.3f}' for p in probs]}")
    
    return {"predictions": final_predictions, "label_ids": final_labels}

model_name = args.model_name_or_path.split("/")[-1]

actual_batch_size = args.batch_size * args.gradient_accumulation_steps
print(f"Actual batch size (batch_size x gradient_accumulation_steps): {actual_batch_size}")

i = 1
# Include PEFT in output directory name
peft_suffix = "lora" if args.use_peft else "full"
output_dir = f"{args.dataset_name_or_path.replace('/', '-')}_{model_name}_{peft_suffix}_{actual_batch_size}_{i}".replace('.', '').replace('_-', '_').replace('-_', '_')

# Check if output_dir exists, if so, increment i
while os.path.exists(output_dir):
    i += 1
    output_dir = f"{args.dataset_name_or_path.replace('/', '-')}_{model_name}_{peft_suffix}_{actual_batch_size}_{i}".replace('.', '').replace('_-', '_').replace('-_', '_')

print(f"Output directory: {output_dir}")

wandb.init(name=output_dir)

# EuroStyle - Adapted from src/euroeval/finetuning.py
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy=IntervalStrategy.STEPS,
    save_strategy=IntervalStrategy.STEPS,
    eval_steps=30,
    logging_steps=30,
    save_steps=30,
    max_steps=args.max_steps,  # Use argument value
    save_total_limit=1,
    per_device_train_batch_size=args.batch_size,  # Default varies
    per_device_eval_batch_size=args.batch_size,
    eval_accumulation_steps=32,
    optim=OptimizerNames.ADAMW_TORCH,
    learning_rate=args.learning_rate,  # EuroEval default is 2e-5
    warmup_ratio=0.01,   # 1% warmup
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    # gradient_checkpointing=True,  # Trade compute for memory
    # dataloader_pin_memory=False,  # Reduce memory usage
    load_best_model_at_end=True,
    push_to_hub=bool(args.push_to_hub),
    hub_model_id=args.hub_model_id,
    report_to="wandb"
)

patience = 20

if USE_CUSTOM_TRAINER:
    TRAINER = MultipleChoiceTrainer
else:
    TRAINER = Trainer

# Initialize custom trainer
trainer = TRAINER(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["val"] if "val" in processed_dataset else processed_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=patience), BestMetricsLoggerCallback()], # EuroStyle
)

# Clear cache before training
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Train the model
trainer.train()

# Evaluate the model with multi-choice metrics
eval_results = trainer.evaluate(eval_dataset=processed_dataset["test"], metric_key_prefix="test")

# Print training arguments and evaluation results
print(f"Training arguments: {training_args}")
print("=" * 60)
print(f"Final evaluation results: {eval_results}")

# Push to hub if requested
if args.push_to_hub:
    trainer.push_to_hub()

print("Training completed!")

"""
Examples:
# PEFT (LoRA) fine-tuning (default)
python src/finetune_multi_choice_cls.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path danske_talemaader

python src/finetune_multi_choice_cls.py --model_name_or_path google/gemma-3-4b-pt --dataset_name_or_path danske_talemaader

# Full fine-tuning (no PEFT)
python src/finetune_multi_choice_cls.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path danske_talemaader --use_peft 0

python src/finetune_multi_choice_cls.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path danish_citizen_test --use_peft 0

python src/finetune_multi_choice_cls.py --model_name_or_path google/gemma-3-4b-pt --dataset_name_or_path danish_citizen_test --use_peft 0

last runs:

python src/finetune_multi_choice_cls.py --model_name_or_path ../new_models/student_step24125_dyna_commonpile --dataset_name_or_path danish_citizen_test --use_peft 0 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path google/gemma-3-4b-pt --dataset_name_or_path danish_citizen_test --use_peft 0 --batch_size 8 --gradient_accumulation_steps 16 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path google/gemma-3-4b-it --dataset_name_or_path danish_citizen_test --use_peft 0 --batch_size 8 --gradient_accumulation_steps 16 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path danish_citizen_test --use_peft 0 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path google/gemma-3-1b-pt --dataset_name_or_path danish_citizen_test --use_peft 0 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path google/gemma-3-1b-it --dataset_name_or_path danish_citizen_test --use_peft 0 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path ../new_models/student_step15908_dyna_none --dataset_name_or_path danish_citizen_test --use_peft 0 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path ../new_models/student_step3678_onpolicy --dataset_name_or_path danish_citizen_test --use_peft 0 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path ../new_models/student_step7356_cos_giga_distill_full --dataset_name_or_path danish_citizen_test --use_peft 0 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path google/gemma-3-4b-pt --dataset_name_or_path danske_talemaader --use_peft 0 --batch_size 8 --gradient_accumulation_steps 16 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path google/gemma-3-4b-it --dataset_name_or_path danske_talemaader --use_peft 0 --batch_size 8 --gradient_accumulation_steps 16 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path google/gemma-3-1b-pt --dataset_name_or_path danske_talemaader --use_peft 0 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path google/gemma-3-1b-it --dataset_name_or_path danske_talemaader --use_peft 0 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path danske_talemaader --use_peft 0 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path ../new_models/student_step15908_dyna_none --dataset_name_or_path danske_talemaader --use_peft 0 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path ../new_models/student_step3678_onpolicy --dataset_name_or_path danske_talemaader --use_peft 0 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path ../new_models/student_step7356_cos_giga_distill_full --dataset_name_or_path danske_talemaader --use_peft 0 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path ../new_models/student_step24125_dyna_commonpile --dataset_name_or_path danske_talemaader --use_peft 0 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path google/gemma-3-4b-pt --dataset_name_or_path hellaswag_da --use_peft 0 --batch_size 8 --gradient_accumulation_steps 16 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path google/gemma-3-4b-it --dataset_name_or_path hellaswag_da --use_peft 0 --batch_size 8 --gradient_accumulation_steps 16 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path google/gemma-3-1b-pt --dataset_name_or_path hellaswag_da --use_peft 0 --batch_size 8 --gradient_accumulation_steps 16 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path google/gemma-3-1b-it --dataset_name_or_path hellaswag_da --use_peft 0 --batch_size 8 --gradient_accumulation_steps 16 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path ../new_models/student_step15908_dyna_none --dataset_name_or_path hellaswag_da --use_peft 0 --batch_size 8 --gradient_accumulation_steps 16 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path ../new_models/student_step7356_cos_giga_distill_full --dataset_name_or_path hellaswag_da --use_peft 0 --batch_size 8 --gradient_accumulation_steps 16 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path ../new_models/student_step7356_cos_giga_distill_full_continual --dataset_name_or_path hellaswag_da --use_peft 0 --batch_size 8 --gradient_accumulation_steps 16 --gpu_device 0

to run:





tests

python src/finetune_multi_choice_cls.py --model_name_or_path google/gemma-3-4b-pt --dataset_name_or_path danske_talemaader --use_peft 0 --batch_size 64 --gpu_device 0

python src/finetune_multi_choice_cls.py --model_name_or_path google/gemma-3-4b-pt --dataset_name_or_path danske_talemaader --use_peft 0 --batch_size 8 --gradient_accumulation_steps 32 --gpu_device 0




python src/finetune_multi_choice_cls.py --model_name_or_path ../new_models/student_step31816_2dyna --dataset_name_or_path danish_citizen_test --use_peft 0 --gpu_device 0 --batch_size 8 --gradient_accumulation_steps 32


"""