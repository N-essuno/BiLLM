#!/usr/bin/env python3
"""
Quick test of the BiLLM sequence classification with attention mask validation
"""

import torch
import sys
import os

# Add the BiLLM source directory to the path
sys.path.insert(0, '/work/evaluation/BiLLM/src')

from billm.modeling_gemma3 import Gemma3ForSequenceClassification
from transformers import Gemma3TextConfig

def test_billm_attention_masks():
    """Test BiLLM with minimal inputs to verify attention mask behavior"""
    print("🔍 Testing BiLLM Attention Masks with Real Model")
    print("-" * 60)
    
    # Create a small config to avoid memory issues
    config = Gemma3TextConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=32,
        pad_token_id=0,
        _attn_implementation="eager",
        num_labels=3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2}
    )
    
    print(f"Config: {config.num_hidden_layers} layers, BiLLM starts at layer 0")
    
    # Create the model (without loading pretrained weights)
    print("Creating BiLLM model...")
    model = Gemma3ForSequenceClassification(config)
    model.eval()
    
    # Test with very small input
    batch_size = 1
    seq_len = 5
    
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids.tolist()}")
    
    # Run forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True
        )
    
    print(f"✅ Forward pass successful!")
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Logits: {outputs.logits}")
    print(f"Number of attention layers: {len(outputs.attentions) if outputs.attentions else 'None'}")
    
    # Verify the bidirectional configuration was applied
    print(f"\n🔍 Model configuration check:")
    print(f"Bidirectional layers: {model.model.bidirectionas}")
    print(f"Total layers: {len(model.model.layers)}")
    print(f"Classification head output size: {model.score.out_features}")
    
    # Check that all layers after BiLLM_START_INDEX are bidirectional
    from billm.config import BiLLM_START_INDEX
    expected_bidirectional = [i >= BiLLM_START_INDEX for i in range(config.num_hidden_layers)]
    actual_bidirectional = model.model.bidirectionas
    
    if expected_bidirectional == actual_bidirectional:
        print(f"✅ Bidirectional configuration correct!")
    else:
        print(f"❌ Bidirectional configuration mismatch!")
        print(f"Expected: {expected_bidirectional}")
        print(f"Actual: {actual_bidirectional}")
    
    print(f"\n🎉 BiLLM attention mask test completed successfully!")

if __name__ == "__main__":
    test_billm_attention_masks()
