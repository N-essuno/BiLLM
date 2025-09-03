#!/usr/bin/env python3
"""
BiLLM Conversion Test Suite
Test script to validate that uni-directional to bi-directional model conversion worked correctly.

Usage:
    python test_billm_conversion.py --model_path /path/to/your/converted/model --model_type llama --task_type causal_lm
    python test_billm_conversion.py --model_path /path/to/your/converted/model --model_type mistral --task_type sequence_classification
    python test_billm_conversion.py --model_path /path/to/your/converted/model --model_type qwen2 --task_type token_classification
    python test_billm_conversion.py --model_path /path/to/your/converted/model --model_type gemma3 --task_type causal_lm

Supported model types: llama, mistral, qwen2, openelm, gemma3
Supported task types: causal_lm, sequence_classification, token_classification
"""

# python test_billm_conversion.py --model_path models/billm_llama_31_8b_laerebogen --model_type llama --task_type sequence_classification
# python test_billm_conversion.py --model_path models/billm_gemma_3_1b_cpt_dynaword_full_v1 --model_type gemma3 --task_type sequence_classification

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer
import traceback
from typing import Dict, Any, Optional, Tuple

# Add src to path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def setup_environment():
    """Setup BiLLM environment for testing"""
    os.environ['BiLLM_START_INDEX'] = '0'  # Enable bidirectional mode
    print("✅ BiLLM environment configured (BiLLM_START_INDEX=0)")


def get_model_class_for_task(model_type: str, task_type: str):
    """Get the appropriate model class based on model type and task type"""
    model_type = model_type.lower()

    # Dynamic import and return the class directly
    try:
        if model_type == 'llama':
            from src.billm import LlamaForCausalLM, LlamaForSequenceClassification, LlamaForTokenClassification
            if task_type == 'causal_lm':
                return LlamaForCausalLM
            elif task_type == 'sequence_classification':
                return LlamaForSequenceClassification
            elif task_type == 'token_classification':
                return LlamaForTokenClassification

        elif model_type == 'mistral':
            from src.billm import MistralForCausalLM, MistralForSequenceClassification, MistralForTokenClassification
            if task_type == 'causal_lm':
                return MistralForCausalLM
            elif task_type == 'sequence_classification':
                return MistralForSequenceClassification
            elif task_type == 'token_classification':
                return MistralForTokenClassification

        elif model_type == 'qwen2':
            from src.billm import Qwen2ForCausalLM, Qwen2ForSequenceClassification, Qwen2ForTokenClassification
            if task_type == 'causal_lm':
                return Qwen2ForCausalLM
            elif task_type == 'sequence_classification':
                return Qwen2ForSequenceClassification
            elif task_type == 'token_classification':
                return Qwen2ForTokenClassification

        elif model_type == 'openelm':
            from src.billm import OpenELMForCausalLM, OpenELMForSequenceClassification, OpenELMForTokenClassification
            if task_type == 'causal_lm':
                return OpenELMForCausalLM
            elif task_type == 'sequence_classification':
                return OpenELMForSequenceClassification
            elif task_type == 'token_classification':
                return OpenELMForTokenClassification

        elif model_type == 'gemma3':
            from src.billm import Gemma3ForCausalLM, Gemma3ForSequenceClassification, Gemma3ForTokenClassification
            if task_type == 'causal_lm':
                return Gemma3ForCausalLM
            elif task_type == 'sequence_classification':
                return Gemma3ForSequenceClassification
            elif task_type == 'token_classification':
                return Gemma3ForTokenClassification

        # If we get here, either model_type or task_type is invalid
        raise ValueError(f"Unsupported model type '{model_type}' or task type '{task_type}' combination")

    except ImportError as e:
        raise ImportError(f"Failed to import model classes for {model_type}: {e}")


def load_model_for_task(model_path: str, model_type: str, task_type: str, **kwargs):
    """Load the appropriate model class for the specified task"""
    ModelClass = get_model_class_for_task(model_type, task_type)

    # Load with task-specific parameters
    if task_type == 'sequence_classification':
        # Default to binary classification if not specified
        num_labels = kwargs.get('num_labels', 2)
        return ModelClass.from_pretrained(model_path, num_labels=num_labels)
    elif task_type == 'token_classification':
        # Default to NER with 9 labels if not specified
        num_labels = kwargs.get('num_labels', 9)  # O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
        return ModelClass.from_pretrained(model_path, num_labels=num_labels)
    else:  # causal_lm
        return ModelClass.from_pretrained(model_path)


def test_model_loading(model_path: str, model_type: str, task_type: str) -> Dict[str, Any]:
    """Test if the model loads correctly and has bidirectional layers"""
    print("\n🔍 Testing Model Loading...")
    results = {"success": False, "error": None, "bidirectional_layers": 0, "task_type": task_type}

    try:
        model = load_model_for_task(model_path, model_type, task_type)

        # Check if model loaded
        if model is None:
            results["error"] = "Model failed to load"
            return results

        # Check bidirectional layers - handle different model structures
        bidirectional_count = 0
        if hasattr(model, 'model') and hasattr(model.model, 'bidirectionas'):
            bidirectional_count = len(model.model.bidirectionas)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'bidirectionas'):
            # Some models might use 'transformer' instead of 'model'
            bidirectional_count = len(model.transformer.bidirectionas)
        elif hasattr(model, 'bidirectionas'):
            # Direct access for some architectures
            bidirectional_count = len(model.bidirectionas)

        results["bidirectional_layers"] = bidirectional_count

        if bidirectional_count > 0:
            results["success"] = True
            print(
                f"✅ {task_type} model ({model_type}) loaded successfully with {bidirectional_count} bidirectional layers")
        else:
            results["error"] = "No bidirectional layers found"
            print("❌ Model loaded but no bidirectional layers found")

    except Exception as e:
        results["error"] = str(e)
        print(f"❌ Model loading failed: {e}")
        traceback.print_exc()

    return results


def test_forward_pass(model_path: str, model_type: str, task_type: str) -> Dict[str, Any]:
    """Test forward pass with dummy input for the specific task"""
    print(f"\n🔍 Testing {task_type.replace('_', ' ').title()} Forward Pass...")
    results = {"success": False, "error": None, "output_shape": None, "task_type": task_type}

    try:
        model = load_model_for_task(model_path, model_type, task_type)
        model.eval()

        # Create dummy input
        batch_size = 2
        seq_length = 32
        vocab_size = model.config.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        results["success"] = True
        results["output_shape"] = list(outputs.logits.shape)

        # Task-specific output validation
        if task_type == 'causal_lm':
            expected_shape = [batch_size, seq_length, vocab_size]
            print(f"✅ Causal LM forward pass successful! Output shape: {outputs.logits.shape}")
        elif task_type == 'sequence_classification':
            expected_shape = [batch_size, model.num_labels]
            print(f"✅ Sequence classification forward pass successful! Output shape: {outputs.logits.shape}")
        elif task_type == 'token_classification':
            expected_shape = [batch_size, seq_length, model.num_labels]
            print(f"✅ Token classification forward pass successful! Output shape: {outputs.logits.shape}")

        # Validate output shape
        if list(outputs.logits.shape) != expected_shape:
            print(f"⚠️  Warning: Expected shape {expected_shape}, got {list(outputs.logits.shape)}")

    except Exception as e:
        results["error"] = str(e)
        print(f"❌ Forward pass failed: {e}")
        traceback.print_exc()

    return results


def test_task_specific_functionality(model_path: str, model_type: str, task_type: str) -> Dict[str, Any]:
    """Test task-specific functionality"""
    print(f"\n🔍 Testing {task_type.replace('_', ' ').title()} Specific Functionality...")
    results = {"success": False, "error": None, "task_type": task_type}

    try:
        model = load_model_for_task(model_path, model_type, task_type)
        model.eval()

        if task_type == 'causal_lm':
            # Test text generation capability
            results.update(test_text_generation(model, model_path))
        elif task_type == 'sequence_classification':
            # Test classification with different sequence lengths
            results.update(test_sequence_classification(model))
        elif task_type == 'token_classification':
            # Test token-level predictions
            results.update(test_token_classification(model))

    except Exception as e:
        results["error"] = str(e)
        print(f"❌ Task-specific functionality test failed: {e}")
        traceback.print_exc()

    return results


def test_text_generation(model, model_path: str) -> Dict[str, Any]:
    """Test text generation for causal LM"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        prompt = "The future of artificial intelligence is"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Test generation with short length to avoid long execution
        with torch.no_grad():
            generated = model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"✅ Text generation successful! Generated: '{generated_text}'")
        return {"success": True, "generated_text": generated_text}

    except Exception as e:
        print(f"❌ Text generation failed: {e}")
        return {"success": False, "error": str(e)}


def test_sequence_classification(model) -> Dict[str, Any]:
    """Test sequence classification with different inputs"""
    try:
        # Test with different sequence lengths
        test_cases = [
            torch.randint(0, model.config.vocab_size, (1, 16)),  # Short sequence
            torch.randint(0, model.config.vocab_size, (1, 64)),  # Medium sequence
            torch.randint(0, model.config.vocab_size, (1, 128)),  # Long sequence
        ]

        predictions = []
        for i, input_ids in enumerate(test_cases):
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions.append(outputs.logits.argmax(dim=-1).item())

        print(f"✅ Sequence classification successful! Predictions: {predictions}")
        return {"success": True, "predictions": predictions}

    except Exception as e:
        print(f"❌ Sequence classification test failed: {e}")
        return {"success": False, "error": str(e)}


def test_token_classification(model) -> Dict[str, Any]:
    """Test token classification predictions"""
    try:
        batch_size = 2
        seq_length = 32
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Get predictions for each token
        predictions = outputs.logits.argmax(dim=-1)

        # Check that we get per-token predictions
        assert predictions.shape == (batch_size,
                                     seq_length), f"Expected shape {(batch_size, seq_length)}, got {predictions.shape}"

        print(f"✅ Token classification successful! Prediction shape: {predictions.shape}")
        return {"success": True, "prediction_shape": list(predictions.shape)}

    except Exception as e:
        print(f"❌ Token classification test failed: {e}")
        return {"success": False, "error": str(e)}


def test_training_step(model_path: str, model_type: str, task_type: str) -> Dict[str, Any]:
    """Test a minimal training step to ensure gradients flow correctly"""
    print(f"\n🔍 Testing {task_type.replace('_', ' ').title()} Training Step...")
    results = {"success": False, "error": None, "loss_value": None, "task_type": task_type}

    model = None
    optimizer = None

    try:
        # Clear GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Smart device selection with fallback
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device")
        else:
            device = torch.device("cpu")
            print("Using CPU device")

        # Load model with memory-efficient settings
        if task_type == 'sequence_classification':
            model = load_model_for_task(model_path, model_type, task_type, num_labels=2)
        elif task_type == 'token_classification':
            model = load_model_for_task(model_path, model_type, task_type, num_labels=9)
        else:  # causal_lm
            model = load_model_for_task(model_path, model_type, task_type)

        # Move model to device with error handling for large models
        try:
            model = model.to(device)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if device.type != "cpu":
                print(f"⚠️ GPU memory insufficient, falling back to CPU. Error: {e}")
                device = torch.device("cpu")
                model = model.to(device)
            else:
                raise e

        model.train()

        # Use only parameters that require gradients for the optimizer
        # For classification tasks, often only the classification head needs training
        if task_type in ['sequence_classification', 'token_classification']:
            # Only train the classification head to save memory
            if hasattr(model, 'classifier'):
                trainable_params = list(model.classifier.parameters())
            elif hasattr(model, 'score'):
                trainable_params = list(model.score.parameters())
            else:
                # Fallback to all parameters
                trainable_params = [p for p in model.parameters() if p.requires_grad]
        else:
            trainable_params = [p for p in model.parameters() if p.requires_grad]

        # Not enough memory to fine-tune all parameters on large models on my mac
        # trainable_params = [p for p in model.parameters() if p.requires_grad]

        print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")

        optimizer = AdamW(trainable_params, lr=1e-5)

        # Dummy batch size and sequence length
        batch_size = 2
        seq_length = 64

        # Set vocab size
        # vocab_size = min(model.config.vocab_size, 32000)  # Cap to reasonable size
        vocab_size = model.config.vocab_size
        print(f"Original vocab_size: {model.config.vocab_size} \nUsing vocab_size: {vocab_size}")

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
        attention_mask = torch.ones_like(input_ids)

        if task_type == 'sequence_classification':
            labels = torch.randint(0, 2, (batch_size,), device=device)  # Binary classification
        elif task_type == 'token_classification':
            labels = torch.randint(0, 9, (batch_size, seq_length), device=device)  # Token-level labels
        else:  # causal_lm
            labels = input_ids.clone()  # Next token prediction

        # Training step with gradient accumulation to save memory
        optimizer.zero_grad()

        # Use gradient checkpointing if available to save memory
        # if hasattr(model, 'gradient_checkpointing_enable'):
        #     model.gradient_checkpointing_enable()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Scale loss if needed
        # loss = loss / 1  # No gradient accumulation for this test
        loss.backward()

        # Clip gradients to prevent explosion
        # torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

        optimizer.step()

        results["success"] = True
        results["loss_value"] = loss.item()
        results["device_used"] = str(device)
        print(f"✅ {task_type.replace('_', ' ').title()} training step successful! Loss: {loss.item():.4f} (Device: {device})")

    except Exception as e:
        results["error"] = str(e)
        print(f"❌ Training step failed: {e}")
        traceback.print_exc()

    finally:
        # Clean up memory
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer

        # Force garbage collection
        import gc
        gc.collect()

        # Clear GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return results


def test_tokenizer_integration(model_path: str, model_type: str, task_type: str) -> Dict[str, Any]:
    """Test with real tokenized text"""
    print(f"\n🔍 Testing {task_type.replace('_', ' ').title()} Tokenizer Integration...")
    results = {"success": False, "error": None, "prediction_shape": None, "task_type": task_type}

    try:
        model = load_model_for_task(model_path, model_type, task_type)
        model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Test texts
        texts = [
            "This is a test sentence for bidirectional processing.",
            "Another example text to validate the conversion worked correctly."
        ]

        # Tokenize
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        results["success"] = True
        results["prediction_shape"] = list(outputs.logits.shape)
        print(
            f"✅ {task_type.replace('_', ' ').title()} tokenizer integration successful! Prediction shape: {outputs.logits.shape}")

    except Exception as e:
        results["error"] = str(e)
        print(f"❌ Tokenizer integration failed: {e}")
        traceback.print_exc()

    return results


def test_batch_processing(model_path: str, model_type: str, task_type: str) -> Dict[str, Any]:
    """Test with different batch sizes to ensure stability"""
    print(f"\n🔍 Testing {task_type.replace('_', ' ').title()} Batch Processing...")
    results = {"success": False, "error": None, "batch_results": {}, "task_type": task_type}

    try:
        model = load_model_for_task(model_path, model_type, task_type)
        model.eval()

        batch_sizes = [1, 4, 8]
        all_passed = True

        for batch_size in batch_sizes:
            try:
                input_ids = torch.randint(0, model.config.vocab_size, (batch_size, 32))
                attention_mask = torch.ones_like(input_ids)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                results["batch_results"][batch_size] = {"success": True, "shape": list(outputs.logits.shape)}
                print(f"✅ Batch size {batch_size}: OK - Output shape: {outputs.logits.shape}")

            except Exception as e:
                results["batch_results"][batch_size] = {"success": False, "error": str(e)}
                print(f"❌ Batch size {batch_size}: Failed - {e}")
                all_passed = False

        results["success"] = all_passed

    except Exception as e:
        results["error"] = str(e)
        print(f"❌ Batch processing test setup failed: {e}")
        traceback.print_exc()

    return results


def test_memory_usage(model_path: str, model_type: str, task_type: str) -> Dict[str, Any]:
    """Test memory usage and cleanup"""
    print(f"\n🔍 Testing {task_type.replace('_', ' ').title()} Memory Usage...")
    results = {"success": False, "error": None, "peak_memory_mb": None, "task_type": task_type}

    try:
        import psutil
        import gc

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        model = load_model_for_task(model_path, model_type, task_type)

        # Run a few forward passes
        for _ in range(3):
            input_ids = torch.randint(0, model.config.vocab_size, (4, 64))
            with torch.no_grad():
                outputs = model(input_ids)

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        # Cleanup
        del model
        gc.collect()

        results["success"] = True
        results["peak_memory_mb"] = round(peak_memory, 2)
        results["memory_increase_mb"] = round(memory_increase, 2)
        print(f"✅ Memory test successful! Peak memory: {peak_memory:.2f} MB (+{memory_increase:.2f} MB)")

    except ImportError:
        results["error"] = "psutil not available for memory monitoring"
        print("⚠️ Memory monitoring skipped (psutil not available)")
    except Exception as e:
        results["error"] = str(e)
        print(f"❌ Memory test failed: {e}")
        traceback.print_exc()

    return results


def generate_summary_report(all_results: Dict[str, Any], model_path: str, model_type: str, task_type: str):
    """Generate a comprehensive test summary"""
    print("\n" + "=" * 80)
    print("📊 BILLM CONVERSION TEST SUMMARY REPORT")
    print("=" * 80)
    print(f"Model Path: {model_path}")
    print(f"Model Type: {model_type.upper()}")
    print(f"Task Type: {task_type.replace('_', ' ').title()}")
    print("-" * 80)

    total_tests = 0
    passed_tests = 0

    for test_name, result in all_results.items():
        total_tests += 1
        if isinstance(result, dict) and result.get("success", False):
            passed_tests += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        print(f"{test_name:<30}: {status}")

        # Print additional details for specific tests
        if test_name == "model_loading" and result.get("bidirectional_layers"):
            print(f"{'':30}   Bidirectional layers: {result['bidirectional_layers']}")
        elif test_name == "batch_processing" and result.get("batch_results"):
            for batch_size, batch_result in result["batch_results"].items():
                batch_status = "✅" if batch_result.get("success") else "❌"
                print(f"{'':30}   Batch {batch_size}: {batch_status}")

    print("-" * 80)
    print(f"Overall Result: {passed_tests}/{total_tests} tests passed")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Test BiLLM conversion results")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the converted BiLLM model")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["llama", "mistral", "qwen2", "openelm", "gemma3"],
                        help="Type of the model (llama, mistral, qwen2, openelm, gemma3)")
    parser.add_argument("--task_type", type=str, required=True,
                        choices=["causal_lm", "sequence_classification", "token_classification"],
                        help="Type of task the model was converted for")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training-related tests")
    parser.add_argument("--skip_memory", action="store_true",
                        help="Skip memory usage tests")
    parser.add_argument("--num_labels", type=int, default=None,
                        help="Number of labels for classification tasks (default: 2 for seq_clf, 9 for token_clf)")

    args = parser.parse_args()

    print("🚀 Starting BiLLM Conversion Test Suite...")
    print(f"Model: {args.model_path}")
    print(f"Type: {args.model_type.upper()}")
    print(f"Task: {args.task_type.replace('_', ' ').title()}")

    # Setup environment
    setup_environment()

    # Run all tests
    all_results = {}

    # Core functionality tests
    all_results["model_loading"] = test_model_loading(args.model_path, args.model_type, args.task_type)
    all_results["forward_pass"] = test_forward_pass(args.model_path, args.model_type, args.task_type)
    all_results["tokenizer_integration"] = test_tokenizer_integration(args.model_path, args.model_type, args.task_type)
    all_results["task_specific_functionality"] = test_task_specific_functionality(args.model_path, args.model_type,
                                                                                  args.task_type)
    all_results["batch_processing"] = test_batch_processing(args.model_path, args.model_type, args.task_type)

    # Optional tests
    if not args.skip_training:
        all_results["training_step"] = test_training_step(args.model_path, args.model_type, args.task_type)

    if not args.skip_memory:
        all_results["memory_usage"] = test_memory_usage(args.model_path, args.model_type, args.task_type)

    # Generate summary
    generate_summary_report(all_results, args.model_path, args.model_type, args.task_type)


if __name__ == "__main__":
    main()
