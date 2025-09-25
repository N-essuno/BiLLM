# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import evaluate
import wandb
from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    EarlyStoppingCallback, 
    TrainerCallback,
    DataCollatorWithPadding,
    TrainingArguments, 
    Trainer
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import IntervalStrategy, EvalPrediction
from transformers.training_args import OptimizerNames
from peft import get_peft_model, LoraConfig, TaskType
from billm import Gemma3ForTokenClassification
from collections import defaultdict
import json
from typing import Dict, List, Tuple, Optional, Any

USE_CUSTOM_TRAINER = True

envs_dir = os.getcwd() + '/envs.env'
print(f"Loading envs from: {envs_dir}")
load_dotenv(envs_dir)

hf_token = os.getenv("HF_TOKEN")
hf_token_euroeval = os.getenv("HF_TOKEN_EUROEVAL")
wandb_api_key = os.getenv("WANDB_API_KEY")

os.environ["WANDB_PROJECT"] = "billm_qa_test"

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True,
                    help='Specify model_name_or_path to set transformer backbone.')
parser.add_argument('--dataset_name_or_path', type=str, required=True,
                    help='Specify huggingface dataset name or local file path.')
parser.add_argument('--epochs', type=int, default=3, help='Specify number of epochs, default 3')
parser.add_argument('--max_steps', type=int, default=5000, help='Specify max steps, default 5000')
parser.add_argument('--batch_size', type=int, default=16, help='Specify batch size, default 16')
parser.add_argument('--learning_rate', type=float, default=3e-5, help='Specify learning rate, default 3e-5')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Specify gradient accumulation steps, default 1')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Specify weight decay, default 0.01')
parser.add_argument('--max_length', type=int, default=512, help='Specify max length, default 512')
parser.add_argument('--max_answer_length', type=int, default=30, help='Maximum answer length in tokens, default 30')
parser.add_argument('--use_peft', type=int, default=1, choices=[0, 1], help='Specify whether to use PEFT (LoRA), default 1 (enabled)')
parser.add_argument('--lora_r', type=int, default=16, help='Specify lora r, default 16')
parser.add_argument('--lora_alpha', type=int, default=32, help='Specify lora alpha, default 32')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='Specify lora dropout, default 0.1')
parser.add_argument('--push_to_hub', type=int, default=0, choices=[0, 1], help='Specify push_to_hub, default 0')
parser.add_argument('--hub_model_id', type=str, default=None, help='Hub model ID for pushing')
parser.add_argument('--gpu_device', type=int, default=None,
                    help='Specify which GPU device to use (0, 1, 2, etc.). If not specified, uses default CUDA device or auto-selects.')

args = parser.parse_args()

# Set CUDA device as early as possible if specified
if args.gpu_device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    torch.cuda.set_device(0)  # After setting CUDA_VISIBLE_DEVICES, device 0 refers to the selected GPU

print(f'Args: {args}')

class BestMetricsLoggerCallback(TrainerCallback):
    def __init__(self):
        self.best_f1 = 0.0
        self.best_em = 0.0
        
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs:
            current_f1 = logs.get('eval_f1', 0.0)
            current_em = logs.get('eval_exact_match', 0.0)
            
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                print(f"🏆 New best F1 score: {self.best_f1:.4f}")
                
            if current_em > self.best_em:
                self.best_em = current_em
                print(f"🏆 New best Exact Match: {self.best_em:.4f}")

def get_special_token_metadata(tokenizer: PreTrainedTokenizerBase) -> dict:
    """Extract special token metadata from the tokenizer."""
    has_cls_token = tokenizer.cls_token is not None
    has_sep_token = tokenizer.sep_token is not None
    cls_token_id = tokenizer.cls_token_id if has_cls_token else None
    cls_token = tokenizer.cls_token if has_cls_token else ""
    sep_token = tokenizer.sep_token if has_sep_token else ""
    
    return {
        "has_cls_token": has_cls_token,
        "has_sep_token": has_sep_token,
        "cls_token_id": cls_token_id,
        "cls_token": cls_token,
        "sep_token": sep_token,
    }

def prepare_train_examples(examples, tokenizer):
    """Prepare the features for training - adapted from EuroEval."""
    # Some of the questions have lots of whitespace on the left, which is not useful
    # and will make the truncation of the context fail (the tokenized question will
    # take a lots of space). So we remove that left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]
    
    # Extract special token metadata from the tokenizer
    special_token_metadata = get_special_token_metadata(tokenizer=tokenizer)
    has_cls_token = special_token_metadata["has_cls_token"]
    has_sep_token = special_token_metadata["has_sep_token"]
    cls_token_id = special_token_metadata["cls_token_id"]
    cls_token = special_token_metadata["cls_token"]
    sep_token = special_token_metadata["sep_token"]

    # If the tokenizer is not adding special tokens, then we add them manually
    if not has_cls_token and not has_sep_token:
        examples["question"] = [f"{cls_token} {q} {sep_token}" for q in examples["question"]]

    # Set the stride used during tokenization, when the context is long enough to be
    # split into several features. Since we are always keeping the question tokens, we
    # need to make sure that the stride does not exceed the resulting maximum context
    # length.
    max_question_tokens = max(len(tokenizer(q).input_ids) for q in examples["question"])
    num_special_tokens = int(has_cls_token) + int(has_sep_token)
    stride = args.max_length // 4
    max_length = args.max_length - stride
    stride = min(stride, max_length - max_question_tokens - num_special_tokens)
    max_length = args.max_length - stride

    # Tokenize our examples with truncation and padding, but keep the overflows using a
    # stride. This results in one example possible giving several features when a
    # context is long, each of those features having a context that overlaps a bit the
    # context of the previous feature.
    tokenized_examples = tokenizer(
        text=examples["question"],
        text_pair=examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we
    # need a map from a feature to its corresponding example. This key gives us just
    # that
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # The offset mappings will give us a map from token to character position in the
    # original context. This will help us compute the start_positions and
    # end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Initialize the start- and end positions of the answers
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id) if tokenizer.cls_token_id in input_ids else 0

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def prepare_test_examples(examples, tokenizer):
    """Prepare test examples - adapted from EuroEval."""
    # Some of the questions have lots of whitespace on the left, which is not useful
    # and will make the truncation of the context fail (the tokenized question will
    # take a lots of space). So we remove that left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Extract special token metadata from the tokenizer
    special_token_metadata = get_special_token_metadata(tokenizer=tokenizer)
    has_cls_token = special_token_metadata["has_cls_token"]
    has_sep_token = special_token_metadata["has_sep_token"]
    cls_token = special_token_metadata["cls_token"]
    sep_token = special_token_metadata["sep_token"]

    # If the tokenizer is not adding special tokens, then we add them manually
    if not has_cls_token and not has_sep_token:
        examples["question"] = [f"{cls_token} {q} {sep_token}" for q in examples["question"]]

    # Set the stride used during tokenization, when the context is long enough to be
    # split into several features. Since we are always keeping the question tokens, we
    # need to make sure that the stride does not exceed the resulting maximum context
    # length.
    max_question_tokens = max(len(tokenizer(q).input_ids) for q in examples["question"])
    num_special_tokens = int(has_cls_token) + int(has_sep_token)
    stride = args.max_length // 4
    max_length = args.max_length - stride
    stride = min(stride, max_length - max_question_tokens - num_special_tokens)
    max_length = args.max_length - stride

    # Tokenize our examples with truncation and maybe padding, but keep the overflows
    # using a stride. This results in one example possible giving several features when
    # a context is long, each of those features having a context that overlaps a bit
    # the context of the previous feature.
    tokenized_examples = tokenizer(
        text=examples["question"],
        text_pair=examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we
    # need a map from a feature to its corresponding example. This key gives us just
    # that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the id that gave us this feature and we will store the offset mappings.
    tokenized_examples["id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path, 
    token=hf_token, 
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load metrics
f1_metric = evaluate.load("squad")

def compute_f1_em(predictions, references):
    """Compute F1 and EM scores for question answering."""
    f1_scores = []
    em_scores = []
    
    for pred, ref in zip(predictions, references):
        # Normalize text for comparison
        pred_text = pred.lower().strip()
        ref_texts = [r.lower().strip() for r in ref] if isinstance(ref, list) else [ref.lower().strip()]
        
        # Exact match
        em = int(pred_text in ref_texts)
        em_scores.append(em)
        
        # F1 score
        if not pred_text or not any(ref_texts):
            f1_scores.append(0.0)
        else:
            # Compute token-level F1
            pred_tokens = pred_text.split()
            max_f1 = 0.0
            
            for ref_text in ref_texts:
                ref_tokens = ref_text.split()
                
                if not pred_tokens and not ref_tokens:
                    max_f1 = 1.0
                    break
                elif not pred_tokens or not ref_tokens:
                    continue
                
                common_tokens = set(pred_tokens) & set(ref_tokens)
                precision = len(common_tokens) / len(pred_tokens)
                recall = len(common_tokens) / len(ref_tokens)
                
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    max_f1 = max(max_f1, f1)
            
            f1_scores.append(max_f1)
    
    return {
        'f1': np.mean(f1_scores),
        'exact_match': np.mean(em_scores)
    }

class QuestionAnsweringTrainer(Trainer):
    """Custom trainer for question answering tasks."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def evaluate(
        self,
        eval_dataset=None,
        orig_eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """Override evaluate method to handle QA-specific evaluation."""
        # Store original dataset for post-processing
        self._orig_eval_dataset = orig_eval_dataset or eval_dataset
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

def postprocess_qa_predictions(
    predictions: Tuple[np.ndarray, np.ndarray],
    features,
    examples,
    tokenizer,
    max_answer_length: int = 30,
    n_best_size: int = 20,
    null_score_diff_threshold: float = 0.0,
):
    """
    Post-process the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.
    """
    if len(predictions) != 2:
        raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")
    
    all_start_logits, all_end_logits = predictions

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = {}
    
    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Loop through all examples
    for example_index, example in enumerate(examples):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Loop through all features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue
                    
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        
        # Only keep the best `n_best_size` predictions.
        predictions_dict = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Add the minimum null prediction
        predictions_dict.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions_dict:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions_dict) == 0 or (len(predictions_dict) == 1 and predictions_dict[0]["text"] == ""):
            predictions_dict.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions_dict])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions_dict.
        for prob, pred in zip(probs, predictions_dict):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not null_score_diff_threshold:
            predictions[example["id"]] = predictions_dict[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions_dict[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions_dict[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = min_null_prediction["score"] - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            if score_diff > null_score_diff_threshold:
                predictions[example["id"]] = ""
            else:
                predictions[example["id"]] = best_non_null_pred["text"]

    return predictions

def compute_metrics(eval_pred: EvalPrediction):
    """Compute QA metrics."""
    predictions, labels = eval_pred
    
    # Get the trainer instance to access datasets
    trainer = compute_metrics._trainer
    
    if hasattr(trainer, '_orig_eval_dataset') and trainer._orig_eval_dataset is not None:
        examples = trainer._orig_eval_dataset
        features = trainer.eval_dataset
        
        # Post-process predictions
        processed_predictions = postprocess_qa_predictions(
            predictions=predictions,
            features=features,
            examples=examples,
            tokenizer=tokenizer,
            max_answer_length=args.max_answer_length,
        )
        
        # Format predictions and references for evaluation
        formatted_predictions = []
        formatted_references = []
        
        for example in examples:
            example_id = example["id"]
            pred_text = processed_predictions.get(example_id, "")
            ref_texts = example["answers"]["text"]
            
            formatted_predictions.append(pred_text)
            formatted_references.append(ref_texts)
        
        # Compute metrics
        return compute_f1_em(formatted_predictions, formatted_references)
    
    # Fallback if we can't access the original dataset
    return {"f1": 0.0, "exact_match": 0.0}

# Load and prepare dataset
print(f"Loading dataset: {args.dataset_name_or_path}")

if args.dataset_name_or_path.startswith("scandiqa_da"):
    dataset = load_dataset("EuroEval/scandiqa-da-mini", token=hf_token_euroeval)
else:
    # Try to load from HuggingFace Hub
    dataset = load_dataset(args.dataset_name_or_path, token=hf_token)

# Ensure we have the required splits
if "train" not in dataset:
    raise ValueError("Dataset must have a 'train' split")

if "val" not in dataset and "test" not in dataset:
    # If no validation set, split train set
    dataset = dataset["train"].train_test_split(test_size=0.1)
    dataset = DatasetDict({
        "train": dataset["train"],
        "validation": dataset["test"]
    })

# Process training data
print("Processing training examples...")
train_dataset = dataset["train"].map(
    lambda examples: prepare_train_examples(examples, tokenizer),
    batched=True,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
)

# Process validation data
print("Processing validation examples...")
eval_split = "val" if "val" in dataset else "test"
eval_dataset = dataset[eval_split].map(
    lambda examples: prepare_test_examples(examples, tokenizer),
    batched=True,
    remove_columns=dataset[eval_split].column_names,
    load_from_cache_file=False,
)

# Initialize model - focusing only on Gemma3 as requested
if any(x in args.model_name_or_path.lower() for x in ['gemma3', 'gemma-3', 'gemma_3']):
    MODEL = Gemma3ForTokenClassification
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
else:
    raise ValueError("This script only supports Gemma3 models. Please use a Gemma3 model.")

# For QA, we need 2 labels: start_position and end_position
num_labels = 2
id2label = {0: "start_position", 1: "end_position"}
label2id = {v: k for k, v in id2label.items()}

model = MODEL.from_pretrained(
    args.model_name_or_path,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    token=hf_token,
)

# Device and dtype handling
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if torch.backends.mps.is_available() and args.gpu_device is None:
    device = torch.device("mps")
    print(f"Using MPS device: {device}")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {device}")
else:
    device = torch.device("cpu")
    print(f"Using CPU device: {device}")

model.to(device)

# Configure LoRA/PEFT if enabled
if args.use_peft:
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
else:
    print("Using full fine-tuning (no LoRA)")

# Data collator for token classification
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

class QuestionAnsweringDataCollator:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    """
    
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        batch = self.tokenizer.pad(
            features,
            padding=True,
            max_length=None,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Handle start and end positions for training
        if "start_positions" in batch:
            batch["start_positions"] = batch["start_positions"].clamp(0)
        if "end_positions" in batch:
            batch["end_positions"] = batch["end_positions"].clamp(0)
            
        return batch

qa_data_collator = QuestionAnsweringDataCollator(tokenizer=tokenizer)

# Store trainer reference in compute_metrics for access to datasets
compute_metrics._trainer = None

model_name = args.model_name_or_path.split("/")[-1]
actual_batch_size = args.batch_size * args.gradient_accumulation_steps
print(f"Actual batch size (batch_size x gradient_accumulation_steps): {actual_batch_size}")

i = 1
peft_suffix = "lora" if args.use_peft else "full"
output_dir = f"{args.dataset_name_or_path.replace('/', '-')}_qa_{model_name}_{peft_suffix}_{actual_batch_size}_{i}".replace('.', '').replace('_-', '_').replace('-_', '_')

# Check if output_dir exists, if so, increment i
while os.path.exists(output_dir):
    i += 1
    output_dir = f"{args.dataset_name_or_path.replace('/', '-')}_qa_{model_name}_{peft_suffix}_{actual_batch_size}_{i}".replace('.', '').replace('_-', '_').replace('-_', '_')

print(f"Output directory: {output_dir}")

wandb.init(name=output_dir)

# Training arguments adapted for QA
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy=IntervalStrategy.STEPS,
    save_strategy=IntervalStrategy.STEPS,
    eval_steps=30,
    logging_steps=30,
    save_steps=30,
    max_steps=args.max_steps,
    num_train_epochs=args.epochs,
    save_total_limit=1,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    eval_accumulation_steps=32,
    optim=OptimizerNames.ADAMW_TORCH,
    learning_rate=args.learning_rate,
    warmup_ratio=0.1,
    # weight_decay=args.weight_decay,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    load_best_model_at_end=True,
    push_to_hub=bool(args.push_to_hub),
    hub_model_id=args.hub_model_id,
    report_to="wandb",
    dataloader_pin_memory=False,  # Avoid memory issues
)

patience = 20

if USE_CUSTOM_TRAINER:
    TRAINER = QuestionAnsweringTrainer
else:
    TRAINER = Trainer

# Initialize trainer
trainer = TRAINER(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=qa_data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=patience), BestMetricsLoggerCallback()],
)

# Store trainer reference for compute_metrics
compute_metrics._trainer = trainer

# Clear cache before training
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Starting training...")
trainer.train()

# Evaluate the model
print("Evaluating model...")
eval_results = trainer.evaluate(
    eval_dataset=eval_dataset, 
    orig_eval_dataset=dataset[eval_split],
    metric_key_prefix="test"
)

# Print results
print(f"Training arguments: {training_args}")
print("=" * 60)
print(f"Final evaluation results: {eval_results}")

# Push to hub if requested
if args.push_to_hub:
    print("Pushing model to hub...")
    trainer.push_to_hub()

print("Training completed!")
