# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
# import evaluate  # Not needed - using custom metrics
import wandb
from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    EarlyStoppingCallback,
    PreTrainedModel, 
    TrainerCallback,
    DataCollatorWithPadding,
    TrainingArguments, 
    Trainer
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import IntervalStrategy, EvalPrediction
from transformers.training_args import OptimizerNames
from peft import get_peft_model, LoraConfig, TaskType
from billm import Gemma3ForQuestionAnswering
from torch.nn import CrossEntropyLoss
from collections import defaultdict
import json
from typing import Dict, List, Tuple, Optional, Any
from datasets.arrow_dataset import Dataset

# NEW
def setup_model_for_question_answering(model: Gemma3ForQuestionAnswering) -> Gemma3ForQuestionAnswering:
    """Setup a model for question answering - adapted from EuroEval."""
    # Get the models' token type embedding children, if they exist
    children = get_children_of_module(name="model", module=model)
    
    if children and isinstance(children, dict):
        # Get the list of attributes that are token type embeddings
        attribute_list = list()
        done = False
        while not done:
            for key, value in children.items():
                attribute_list.append(key)
                if isinstance(value, dict):
                    children = value
                else:
                    done = True
                break

        # Get the token type embeddings if they exist
        if attribute_list:
            try:
                token_type_embeddings = model
                for attribute in attribute_list:
                    token_type_embeddings = getattr(token_type_embeddings, attribute)
                
                if hasattr(token_type_embeddings, 'weight'):
                    token_type_embedding_tensor = token_type_embeddings.weight.data
                    
                    # If the token type embeddings has shape (1, ...) then set the shape to
                    # (2, ...) by randomly initializing the second token type embedding
                    if token_type_embedding_tensor.shape[0] == 1:
                        token_type_embeddings.weight.data = torch.cat(
                            (
                                token_type_embedding_tensor,
                                torch.rand_like(token_type_embedding_tensor),
                            ),
                            dim=0,
                        )
                        token_type_embeddings.num_embeddings = 2
                        
                    # Set the model config to use the new type vocab size
                    model.config.type_vocab_size = 2
            except AttributeError:
                # Model doesn't have token type embeddings, which is fine for Gemma3
                pass
    
    return model
# NEW
def get_children_of_module(name: str, module: nn.Module) -> nn.Module | dict[str, Any] | None:
    """Get the children of a module - adapted from EuroEval."""
    if len(list(module.children())) == 0:
        if name == "token_type_embeddings":
            return module
        else:
            return None
    else:
        submodules = dict()
        for subname, submodule in module.named_children():
            children = get_children_of_module(name=subname, module=submodule)
            if children:
                submodules[subname] = children
        return submodules

USE_CUSTOM_TRAINER = True

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

# Set CUDA device as early as possible if specified - following multi-choice pattern
if args.gpu_device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    torch.cuda.set_device(0)  # After setting CUDA_VISIBLE_DEVICES, device 0 refers to the selected GPU

print(f'Args: {args}')

class BestMetricsLoggerCallback(TrainerCallback):    
    def __init__(self):
        self.best_metrics = {}
        self.metric_keywords = [
            "precision", "recall", "matthews_correlation", "mcc", "accuracy", "loss", "f1", "exact_match"
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

        # Manually ensure that the special tokens are set to None in `sequence_ids` (following EuroEval)
        for special_token in tokenizer.special_tokens_map.keys():
            if hasattr(tokenizer, f"{special_token}_id"):
                special_token_id = getattr(tokenizer, f"{special_token}_id")
                if special_token_id is not None:
                    sequence_ids = [
                        None if token_id == special_token_id else seq_id
                        for token_id, seq_id in zip(input_ids, sequence_ids)
                    ]

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
                while (
                    token_start_index <= token_end_index
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                token_start_index -= 1
                tokenized_examples["start_positions"].append(token_start_index)
                while (
                    token_start_index <= token_end_index
                    and offsets[token_end_index][1] >= end_char
                ):
                    token_end_index -= 1
                token_end_index += 1
                tokenized_examples["end_positions"].append(token_end_index)
                # Add assertion to match EuroEval
                assert token_end_index >= token_start_index

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

# Load metrics - using built-in computation instead of evaluate library
# f1_metric = evaluate.load("squad")  # Not needed - using custom metrics

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




# class QuestionAnsweringTrainer(Trainer):
#     """Trainer subclass for question answering tasks."""

#     def __init__(
#         self,
#         model: "PreTrainedModel | nn.Module",
#         processing_class: "PreTrainedTokenizerBase",
#         args: "TrainingArguments",
#         train_dataset: "Dataset",
#         eval_dataset: "Dataset",
#         compute_metrics: "c.Callable[[EvalPrediction], dict[str, float]]",
#         callbacks: "list[TrainerCallback]",
#         data_collator: "c.Callable",
#         **kwargs,
#     ) -> None:
#         """Initialise the trainer."""
#         super().__init__(
#             model=model,
#             processing_class=processing_class,
#             args=args,
#             train_dataset=train_dataset,
#             eval_dataset=eval_dataset,
#             compute_metrics=compute_metrics,
#             callbacks=callbacks,
#             data_collator=data_collator,
#             **kwargs,
#         )

#         # Get the CLS token id for the tokeniser
#         if self.tokenizer is not None:
#             assert isinstance(self.tokenizer, PreTrainedTokenizerBase)
#             special_token_metadata = get_special_token_metadata(self.tokenizer)
#             self.cls_token_id = special_token_metadata["cls_token_id"]

#         # Set the label names
#         self.label_names = ["start_positions", "end_positions"]

#     def evaluate(  # type: ignore[override]
#         self,
#         eval_dataset: "Dataset | None" = None,
#         orig_eval_dataset: "Dataset | None" = None,
#         ignore_keys: list[str] | None = None,
#         metric_key_prefix: str = "eval",
#     ) -> dict[str, float]:
#         """Evaluate the model on the given dataset.

#         Args:
#             eval_dataset:
#                 The dataset to evaluate on. If None, then use the stored evaluation
#                 dataset.
#             orig_eval_dataset:
#                 The original evaluation dataset, before any postprocessing. If None,
#                 then use the stored original evaluation dataset.
#             ignore_keys:
#                 The keys to ignore when computing the metrics.
#             metric_key_prefix:
#                 The prefix to use for the metric keys.

#         Returns:
#             The metrics computed on the evaluation dataset.
#         """
#         eval_dataloader = self.get_eval_dataloader(eval_dataset)

#         # Temporarily disable metric computation, we will do it in the loop here.
#         compute_metrics = self.compute_metrics  # type: ignore[has-type]
#         self.compute_metrics = None
#         eval_loop = (
#             self.prediction_loop
#             if self.args.use_legacy_prediction_loop
#             else self.evaluation_loop
#         )
#         try:
#             output = eval_loop(
#                 eval_dataloader,
#                 description="Evaluation",
#                 prediction_loss_only=True if compute_metrics is None else None,
#                 ignore_keys=ignore_keys,
#                 metric_key_prefix=metric_key_prefix,
#             )
#         finally:
#             self.compute_metrics = compute_metrics

#         predictions = output.predictions
#         assert isinstance(predictions, tuple)

#         metrics = output.metrics
#         assert metrics is not None

#         if orig_eval_dataset is not None:
#             preds_and_labels = postprocess_predictions_and_labels(
#                 predictions=predictions,  # type: ignore[arg-type]
#                 dataset=orig_eval_dataset,
#                 prepared_dataset=eval_dataset,
#                 cls_token_index=self.cls_token_id,
#             )
#             assert self.compute_metrics is not None
#             new_metrics = self.compute_metrics(preds_and_labels)  # type: ignore[arg-type]
#             metrics.update(new_metrics)

#             # Prefix all keys with metric_key_prefix + '_'
#             for key in list(metrics.keys()):
#                 if not key.startswith(f"{metric_key_prefix}_"):
#                     metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

#         # Only the main node log the results by default
#         if self.args.should_log:
#             self.log(metrics)

#         self.control = self.callback_handler.on_evaluate(
#             self.args,
#             self.state,
#             self.control,  # type: ignore[has-type]
#             metrics,
#         )
#         return metrics




class QuestionAnsweringTrainer(Trainer):
    """Custom trainer for question answering tasks with proper post-processing."""
    
    def __init__(self, *args, orig_eval_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Store original evaluation dataset for post-processing
        self._orig_eval_dataset = orig_eval_dataset
        
        # Set trainer reference in compute_metrics for access to datasets
        if self.compute_metrics is not None:
            self.compute_metrics._trainer = self
        
                # Get the CLS token id for the tokeniser
        if self.tokenizer is not None:
            assert isinstance(self.tokenizer, PreTrainedTokenizerBase)
            special_token_metadata = get_special_token_metadata(self.tokenizer)
            self.cls_token_id = special_token_metadata["cls_token_id"]

        print(f"CLS token ID: {self.cls_token_id}")

        cls_token_id = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else 0

        print(f"CLS token ID 2: {cls_token_id}")

        exit(0)
    

    def evaluate(  # type: ignore[override]
        self,
        eval_dataset: "Dataset | None" = None,
        orig_eval_dataset: "Dataset | None" = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Evaluate the model on the given dataset.

        Args:
            eval_dataset:
                The dataset to evaluate on. If None, then use the stored evaluation
                dataset.
            orig_eval_dataset:
                The original evaluation dataset, before any postprocessing. If None,
                then use the stored original evaluation dataset.
            ignore_keys:
                The keys to ignore when computing the metrics.
            metric_key_prefix:
                The prefix to use for the metric keys.

        Returns:
            The metrics computed on the evaluation dataset.
        """
        eval_dataset = self.eval_dataset
        orig_eval_dataset = self._orig_eval_dataset

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics  # type: ignore[has-type]
        self.compute_metrics = None
        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        predictions = output.predictions
        assert isinstance(predictions, tuple)

        metrics = output.metrics
        assert metrics is not None

        if orig_eval_dataset is not None:
            preds_and_labels = postprocess_predictions_and_labels(
                predictions=predictions,  # type: ignore[arg-type]
                dataset=orig_eval_dataset,
                prepared_dataset=eval_dataset,
                cls_token_index=self.cls_token_id,
            )
            assert self.compute_metrics is not None
            new_metrics = self.compute_metrics(preds_and_labels)  # type: ignore[arg-type]
            metrics.update(new_metrics)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        # Only the main node log the results by default
        if self.args.should_log:
            self.log(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,  # type: ignore[has-type]
            metrics,
        )

        print("======== Evaluation Results ========")
        print(metrics)
        print("====================================")

        exit(0)

        return metrics

    
    # def evaluate(self, eval_dataset=None, orig_eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
    #     orig_eval_dataset = self._orig_eval_dataset
    #     eval_dataset = self.eval_dataset

    #     print("======== params in evaluate ========")
    #     print(eval_dataset)
    #     print(orig_eval_dataset)
    #     print(ignore_keys)
    #     print("===============================")

    #     """Evaluate the model with proper loss computation."""
    #     # Use the parent class evaluate method to get baseline metrics including loss
    #     eval_results = super().evaluate(
    #         eval_dataset=eval_dataset,
    #         ignore_keys=ignore_keys,
    #         metric_key_prefix=metric_key_prefix
    #     )

    #     # Now add our custom QA metrics
    #     if hasattr(self, '_orig_eval_dataset') and self.compute_metrics is not None:
    #         eval_examples = orig_eval_dataset if orig_eval_dataset is not None else self._orig_eval_dataset
            
    #         if eval_examples is not None:
    #             # Get predictions from the model
    #             eval_dataloader = self.get_eval_dataloader(eval_dataset)
                
    #             # Temporarily disable metric computation to get raw predictions
    #             compute_metrics = self.compute_metrics
    #             self.compute_metrics = None
                
    #             try:
    #                 eval_loop = (
    #                     self.prediction_loop
    #                     if self.args.use_legacy_prediction_loop
    #                     else self.evaluation_loop
    #                 )
    #                 output = eval_loop(
    #                     eval_dataloader,
    #                     description="Evaluation",
    #                     prediction_loss_only=False,
    #                     ignore_keys=ignore_keys,
    #                     metric_key_prefix=metric_key_prefix,
    #                 )
    #             finally:
    #                 self.compute_metrics = compute_metrics
                
    #             # Get QA-specific metrics
    #             if output.predictions is not None:
    #                 eval_pred = EvalPrediction(predictions=output.predictions, label_ids=None)
    #                 qa_metrics = self.compute_metrics(eval_pred)
                    
    #                 # Add QA metrics with proper prefix
    #                 for key, value in qa_metrics.items():
    #                     prefixed_key = f"{metric_key_prefix}_{key}" if not key.startswith(f"{metric_key_prefix}_") else key
    #                     eval_results[prefixed_key] = value

    #     # Print detailed results
    #     print(f"\n=== Question Answering Evaluation Results ({metric_key_prefix}) ===")
    #     print(f"Evaluation loss: {eval_results.get(f'{metric_key_prefix}_loss', 'N/A')}")
    #     print(f"QA F1 score: {eval_results.get(f'{metric_key_prefix}_f1', 'N/A')}")
    #     print(f"QA Exact Match: {eval_results.get(f'{metric_key_prefix}_exact_match', 'N/A')}")
    #     print("=" * (len(metric_key_prefix) + 50) + "\n")

    #     return eval_results
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss for QA model - ensures evaluation loss is available."""
        # Let the model compute loss naturally first
        outputs = model(**inputs)
        
        # If model already computed loss, use it (this should work for training)
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        else:
            # Only compute manually if model didn't provide loss (evaluation edge case)
            start_positions = inputs.get("start_positions")
            end_positions = inputs.get("end_positions")
            
            if start_positions is not None and end_positions is not None:
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2
            else:
                # No positions provided during evaluation - set loss to 0
                loss = torch.tensor(0.0, device=outputs.start_logits.device, requires_grad=False)
        
        return (loss, outputs) if return_outputs else loss


def postprocess_predictions_and_labels(
    predictions: tuple[np.ndarray, ...],
    dataset: "Dataset",
    prepared_dataset: "Dataset",
    cls_token_index: int,
) -> tuple[list[dict], list[dict]]:
    """Postprocess the predictions and labels, to allow easier metric computation.

    Args:
        predictions:
            A tuple whose first two elements are (start_logits, end_logits).
        dataset:
            The dataset containing the examples.
        prepared_dataset:
            The dataset containing the prepared examples.
        cls_token_index:
            The index of the CLS token.

    Returns:
        The postprocessed predictions and labels.
    """
    if len(predictions) < 2:
        raise ValueError("`predictions` should be a tuple with at least two elements.")

    all_start_logits, all_end_logits = predictions[:2]

    # Build a map from an example to its corresponding features, being the blocks of
    # text from the context that we're feeding into the model. An example can have
    # multiple features/blocks if it has a long context.
    id_to_index = {k: i for i, k in enumerate(dataset["id"])}
    features_per_example = defaultdict(list)
    for i, feature in enumerate(prepared_dataset):
        id = feature["id"]
        example_index = id_to_index[id]
        features_per_example[example_index].append(i)

    # Loop over all the examples
    prediction_list: list[dict[str, t.Any]] = list()
    labels = list()
    for example_index, example in enumerate(dataset):
        # Extract the best valid answer associated with the current example
        best_answer = find_best_answer(
            all_start_logits=all_start_logits,
            all_end_logits=all_end_logits,
            prepared_dataset=prepared_dataset,
            feature_indices=features_per_example[example_index],
            context=example["context"],
            max_answer_length=30,
            num_best_logits=20,
            min_null_score=0.0,
            cls_token_index=cls_token_index,
        )

        # Create the final prediction dictionary, to be added to the list of
        # predictions
        prediction = dict(
            id=example["id"], prediction_text=best_answer, no_answer_probability=0.0
        )

        # Add the answer to the list of predictions
        prediction_list.append(prediction)

        # Create the associated reference dictionary, to be added to the list of
        # references
        label = dict(
            id=example["id"],
            answers=dict(
                text=example["answers"]["text"],
                answer_start=example["answers"]["answer_start"],
            ),
        )

        # Add the answer and label to the list of predictions and labels, respectively
        labels.append(label)

    return prediction_list, labels

def find_best_answer(
    all_start_logits: np.ndarray,
    all_end_logits: np.ndarray,
    prepared_dataset: "Dataset",
    feature_indices: list[int],
    context: str,
    max_answer_length: int,
    num_best_logits: int,
    min_null_score: float,
    cls_token_index: int,
) -> str:
    """Find the best answer for a given example.

    Args:
        all_start_logits:
            The start logits for all the features.
        all_end_logits:
            The end logits for all the features.
        prepared_dataset:
            The dataset containing the prepared examples.
        feature_indices:
            The indices of the features associated with the current example.
        context:
            The context of the example.
        max_answer_length:
            The maximum length of the answer.
        num_best_logits:
            The number of best logits to consider.
        min_null_score:
            The minimum score an answer can have.
        cls_token_index:
            The index of the CLS token.

    Returns:
        The best answer for the example.
    """
    # Loop through all the features associated to the current example
    valid_answers = list()
    for feature_index in feature_indices:
        # Get the features associated with the current example
        features = prepared_dataset[feature_index]

        # Get the predictions of the model for this feature
        start_logits = all_start_logits[feature_index]
        end_logits = all_end_logits[feature_index]

        # Update minimum null prediction
        cls_index = features["input_ids"].index(cls_token_index)
        feature_null_score = (start_logits[cls_index] + end_logits[cls_index]).item()
        if min_null_score < feature_null_score:
            min_null_score = feature_null_score

        # Find the valid answers for the feature
        valid_answers_for_feature = find_valid_answers(
            start_logits=start_logits,
            end_logits=end_logits,
            offset_mapping=features["offset_mapping"],
            context=context,
            max_answer_length=max_answer_length,
            num_best_logits=num_best_logits,
            min_null_score=min_null_score,
        )
        valid_answers.extend(valid_answers_for_feature)

    # In the very rare edge case we have not a single non-null prediction, we create a
    # fake prediction to avoid failure
    if not valid_answers:
        return ""

    # Otherwise, we select the answer with the largest score as the best answer, and
    # return it
    best_answer_dict = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
    return best_answer_dict["text"]

def find_valid_answers(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    offset_mapping: list[tuple[int, int]],
    context: str,
    max_answer_length: int,
    num_best_logits: int,
    min_null_score: float,
) -> list[dict]:
    """Find the valid answers from the start and end indexes.

    Args:
        start_logits:
            The logits for the start of the answer.
        end_logits:
            The logits for the end of the answer.
        offset_mapping:
            The offset mapping, being a list of pairs of integers for each token index,
            containing the start and end character index in the original context.
        context:
            The context of the example.
        max_answer_length:
            The maximum length of the answer.
        num_best_logits:
            The number of best logits to consider. Note that this function will run in
            O(`num_best_logits` ^ 2) time.
        min_null_score:
            The minimum score an answer can have.

    Returns:
        A list of the valid answers, each being a dictionary with keys "text" and
        "score", the score being the sum of the start and end logits.
    """
    # Fetch the top-k predictions for the start- and end token indices
    start_indexes = np.argsort(start_logits)[-1 : -num_best_logits - 1 : -1].tolist()
    end_indexes = np.argsort(end_logits)[-1 : -num_best_logits - 1 : -1].tolist()

    # We loop over all combinations of starting and ending indexes for valid answers
    valid_answers = list()
    for start_index in start_indexes:
        for end_index in end_indexes:
            # If the starting or ending index is out-of-scope, meaning that they are
            # either out of bounds or correspond to part of the input_ids that are not
            # in the context, then we skip this index
            if (
                start_index >= len(offset_mapping)
                or end_index >= len(offset_mapping)
                or tuple(offset_mapping[start_index]) == (-1, -1)
                or tuple(offset_mapping[end_index]) == (-1, -1)
            ):
                continue

            # Do not consider answers with a length that is either negative or greater
            # than the context length
            max_val = max_answer_length + start_index - 1
            if end_index < start_index or end_index > max_val:
                continue

            # If we got to this point then the answer is valid, so we store the
            # corresponding start- and end character indices in the original context,
            # and from these extract the answer
            start_char = offset_mapping[start_index][0]
            end_char = offset_mapping[end_index][1]
            text = context[start_char:end_char]

            # Compute the score of the answer, being the sum of the start and end
            # logits. Intuitively, this indicates how likely the answer is to be
            # correct, and allows us to pick the best valid answer.
            score = start_logits[start_index] + end_logits[end_index]

            # Add the answer to the list of valid answers, if the score is greater
            # than the minimum null score
            if score > min_null_score:
                valid_answers.append(dict(score=score, text=text))

    return valid_answers

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
    """Compute QA metrics with proper post-processing - matches EuroEval pattern."""
    predictions, labels = eval_pred
    
    # Get the trainer and datasets from the global reference
    if not hasattr(compute_metrics, '_trainer') or compute_metrics._trainer is None:
        # If trainer not available, return dummy metrics
        return {"f1": 0.0, "exact_match": 0.0}
    
    trainer = compute_metrics._trainer
    
    # Use the evaluation dataset and original examples
    if hasattr(trainer, '_orig_eval_dataset') and hasattr(trainer, 'eval_dataset'):
        eval_examples = trainer._orig_eval_dataset
        eval_features = trainer.eval_dataset
        
        # Get CLS token index for postprocessing (following EuroEval pattern)
        cls_token_id = trainer.tokenizer.cls_token_id if trainer.tokenizer.cls_token_id is not None else 0
        
        # Use EuroEval-style postprocessing
        from collections import defaultdict
        
        # Build a map from an example to its corresponding features
        id_to_index = {k: i for i, k in enumerate(eval_examples["id"])}
        features_per_example = defaultdict(list)
        for i, feature in enumerate(eval_features):
            feature_id = feature["id"]
            example_index = id_to_index[feature_id]
            features_per_example[example_index].append(i)
        
        # Extract predictions for each example
        formatted_predictions = []
        formatted_references = []
        
        all_start_logits, all_end_logits = predictions
        
        for example_index, example in enumerate(eval_examples):
            # Find best answer using EuroEval-style logic
            feature_indices = features_per_example[example_index]
            
            # Get the best answer for this example
            best_answer = find_best_answer_euroeval_style(
                all_start_logits=all_start_logits,
                all_end_logits=all_end_logits,
                eval_features=eval_features,
                feature_indices=feature_indices,
                context=example["context"],
                cls_token_id=cls_token_id,
            )
            
            formatted_predictions.append(best_answer)
            formatted_references.append(example["answers"]["text"])
        
        # Compute F1 and EM scores
        metrics = compute_f1_em(formatted_predictions, formatted_references)
        return metrics
    else:
        # Fallback to dummy metrics if datasets not properly stored
        return {"f1": 0.0, "exact_match": 0.0}

def find_best_answer_euroeval_style(
    all_start_logits: np.ndarray,
    all_end_logits: np.ndarray,
    eval_features,
    feature_indices: List[int],
    context: str,
    cls_token_id: int,
    max_answer_length: int = 30,
    num_best_logits: int = 20,
) -> str:
    """Find the best answer following EuroEval's exact logic."""
    valid_answers = []
    min_null_score = -1000000.0  # Very low initial score
    
    # Loop through all features for this example
    for feature_index in feature_indices:
        # Get the predictions for this feature
        start_logits = all_start_logits[feature_index]
        end_logits = all_end_logits[feature_index]
        
        # Get feature data
        feature = eval_features[feature_index]
        input_ids = feature["input_ids"]
        
        # Find CLS token index
        try:
            cls_index = input_ids.index(cls_token_id)
        except ValueError:
            cls_index = 0
        
        # Update minimum null score
        feature_null_score = (start_logits[cls_index] + end_logits[cls_index]).item()
        if min_null_score < feature_null_score:
            min_null_score = feature_null_score
        
        # Get offset mapping
        offset_mapping = feature["offset_mapping"]
        
        # Find valid answers for this feature
        valid_answers_for_feature = find_valid_answers_euroeval_style(
            start_logits=start_logits,
            end_logits=end_logits,
            offset_mapping=offset_mapping,
            context=context,
            max_answer_length=max_answer_length,
            num_best_logits=num_best_logits,
            min_null_score=min_null_score,
        )
        valid_answers.extend(valid_answers_for_feature)
    
    # Return best answer or empty string
    if not valid_answers:
        return ""
    
    best_answer_dict = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
    return best_answer_dict["text"]

def find_valid_answers_euroeval_style(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    offset_mapping: list,
    context: str,
    max_answer_length: int,
    num_best_logits: int,
    min_null_score: float,
) -> List[dict]:
    """Find valid answers following EuroEval's exact logic."""
    # Get top-k predictions for start and end
    start_indexes = np.argsort(start_logits)[-1 : -num_best_logits - 1 : -1].tolist()
    end_indexes = np.argsort(end_logits)[-1 : -num_best_logits - 1 : -1].tolist()
    
    valid_answers = []
    for start_index in start_indexes:
        for end_index in end_indexes:
            # Check if indices are valid
            if (
                start_index >= len(offset_mapping)
                or end_index >= len(offset_mapping)
                or offset_mapping[start_index] is None
                or offset_mapping[end_index] is None
            ):
                continue
            
            # Check answer length constraints
            max_val = max_answer_length + start_index - 1
            if end_index < start_index or end_index > max_val:
                continue
            
            # Extract answer text
            start_char = offset_mapping[start_index][0]
            end_char = offset_mapping[end_index][1]
            text = context[start_char:end_char]
            
            # Calculate score
            score = start_logits[start_index] + end_logits[end_index]
            
            # Add if score is good enough
            if score > min_null_score:
                valid_answers.append({"score": score, "text": text})
    
    return valid_answers

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
        "val": dataset["test"]
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
    MODEL = Gemma3ForQuestionAnswering
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
else:
    raise ValueError("This script only supports Gemma3 models. Please use a Gemma3 model.")

model = MODEL.from_pretrained(
    args.model_name_or_path,
    token=hf_token,
)

# Setup model for QA following EuroEval pattern
model = setup_model_for_question_answering(model)

# Device and dtype handling - following multi-choice pattern
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
        task_type=TaskType.QUESTION_ANS,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
else:
    print("Using full fine-tuning (no PEFT)...")
    # For full fine-tuning, all parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} || All params: {total_params:,} || Trainable%: {100 * trainable_params / total_params:.2f}")

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
    # metric_for_best_model="eval_loss",  # Use eval_loss for model selection
    # greater_is_better=False,  # Lower loss is better
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
if USE_CUSTOM_TRAINER:
    # Pass original evaluation dataset for post-processing
    trainer = TRAINER(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=qa_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience), BestMetricsLoggerCallback()],
        orig_eval_dataset=dataset[eval_split],  # Pass original dataset for post-processing
    )
else:
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

# Clear cache before training
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

print("Starting training...")

trainer.train()

# Evaluate the model
print("Evaluating model...")
eval_results = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="test")

# Print results
print(f"Training arguments: {training_args}")
print("=" * 60)
print(f"Final evaluation results: {eval_results}")

# Push to hub if requested
if args.push_to_hub:
    print("Pushing model to hub...")
    trainer.push_to_hub()

print("Training completed!")

"""




to run:

python src/finetune_question_answering.py --model_name_or_path google/gemma-3-1b-pt --dataset_name_or_path scandiqa_da --use_peft 0 --batch_size 8 --gradient_accumulation_steps 16 --gpu_device 1

"""
