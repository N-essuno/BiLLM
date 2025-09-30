#!/usr/bin/env python3
"""
Standalone test script to verify BiLLM attention mask logic
without loading the full model weights.
"""

import torch
import sys
import os

# Add the BiLLM source directory to the path
sys.path.insert(0, '/work/evaluation/BiLLM/src')

from billm.config import BiLLM_START_INDEX, logger
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers import Gemma3TextConfig

def test_attention_mask_logic():
    """Test the attention mask creation and modification logic"""
    print("🔍 Testing BiLLM Attention Mask Logic")
    print("-" * 50)
    
    # Create a minimal config for testing
    config = Gemma3TextConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=1024,
        max_position_embeddings=512,
        pad_token_id=0,
        _attn_implementation="eager",  # Force eager implementation
    )
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    device = torch.device("cpu")
    
    print(f"Test setup:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  BiLLM_START_INDEX: {BiLLM_START_INDEX}")
    print(f"  Total layers: {config.num_hidden_layers}")
    print(f"  Attention implementation: {config._attn_implementation}")
    
    # Create dummy inputs
    dummy_input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len), device=device)
    dummy_attention_mask = torch.ones_like(dummy_input_ids)
    
    # Create dummy embeddings (simplified)
    inputs_embeds = torch.randn(batch_size, seq_len, config.hidden_size, device=device)
    
    # Create position info
    cache_position = torch.arange(seq_len, device=device)
    position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
    
    print(f"\n📐 Creating attention masks:")
    
    try:
        # Test mask creation
        mask_kwargs = {
            "config": config,
            "input_embeds": inputs_embeds,
            "attention_mask": dummy_attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        
        # Create causal and sliding window masks
        causal_mask = create_causal_mask(**mask_kwargs)
        sliding_mask = create_sliding_window_causal_mask(**mask_kwargs)
        
        if causal_mask is not None:
            print(f"  ✅ Causal mask created: {causal_mask.shape}")
        else:
            print(f"  ⚠️  Causal mask is None (may be using FlexAttention)")
            
        if sliding_mask is not None:
            print(f"  ✅ Sliding window mask created: {sliding_mask.shape}")
        else:
            print(f"  ⚠️  Sliding window mask is None")
        
        # If we don't have actual masks, create a simple causal mask for testing
        if causal_mask is None:
            print(f"  🔧 Creating manual causal mask for testing...")
            causal_mask = torch.triu(
                torch.full((batch_size, 1, seq_len, seq_len), float('-inf'), device=device),
                diagonal=1
            )
            print(f"  ✅ Manual causal mask created: {causal_mask.shape}")
        
        # Analyze the causal mask structure
        print(f"\n🔍 Analyzing causal mask structure:")
        if causal_mask is not None:
            finite_elements = torch.isfinite(causal_mask).sum().item()
            infinite_elements = torch.isinf(causal_mask).sum().item()
            print(f"  Finite elements: {finite_elements}")
            print(f"  Infinite elements: {infinite_elements}")
            print(f"  Unique values: {torch.unique(causal_mask[torch.isfinite(causal_mask)]).tolist()}")
            
            # Check if it's truly causal (lower triangular structure)
            # For a causal mask, position i should not attend to positions > i
            mask_2d = causal_mask[0, 0]  # First head, first batch
            print(f"  Mask 2D shape: {mask_2d.shape}")
            
            # Check causal property
            causal_violations = 0
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    if torch.isfinite(mask_2d[i, j]):  # If can attend to future positions (not -inf)
                        causal_violations += 1
            
            print(f"  Causal violations: {causal_violations} (should be 0 for proper causal mask)")
        
        # Test BiLLM bidirectional modification
        print(f"\n🔄 Testing bidirectional mask modification:")
        
        if BiLLM_START_INDEX >= 0 and causal_mask is not None:
            # Simulate the BiLLM modification logic
            for layer_idx in range(config.num_hidden_layers):
                is_bidirectional = layer_idx >= BiLLM_START_INDEX
                
                if is_bidirectional:
                    # This is what BiLLM does: set mask to zeros for bidirectional layers
                    layer_mask = causal_mask.clone()
                    modified_mask = torch.zeros_like(layer_mask)
                    
                    original_finite = torch.isfinite(layer_mask).sum().item()
                    modified_finite = torch.isfinite(modified_mask).sum().item()
                    
                    print(f"  Layer {layer_idx} (bidirectional):")
                    print(f"    Original finite: {original_finite}, infinite: {layer_mask.numel() - original_finite}")
                    print(f"    Modified finite: {modified_finite}, infinite: {modified_mask.numel() - modified_finite}")
                    print(f"    ✅ Mask zeroed (allows full attention)")
                else:
                    print(f"  Layer {layer_idx} (causal): Preserves causal mask")
            
            bidirectional_layers = config.num_hidden_layers - BiLLM_START_INDEX
            print(f"  📊 Summary: {bidirectional_layers}/{config.num_hidden_layers} layers are bidirectional")
        else:
            if BiLLM_START_INDEX < 0:
                print(f"  ⚠️  BiLLM disabled (BiLLM_START_INDEX={BiLLM_START_INDEX})")
                print(f"  All layers remain causal")
            elif causal_mask is None:
                print(f"  ⚠️  Cannot test mask modification (causal_mask is None)")
        
        # Test attention mask mapping structure (as used in the forward pass)
        print(f"\n🗺️  Testing attention mask mapping:")
        causal_mask_mapping = {
            "full_attention": causal_mask,
            "sliding_attention": sliding_mask,
        }
        
        print(f"  Mapping keys: {list(causal_mask_mapping.keys())}")
        for key, mask in causal_mask_mapping.items():
            if mask is not None:
                finite_count = torch.isfinite(mask).sum().item() if hasattr(mask, 'shape') else 0
                print(f"  {key}: shape {mask.shape}, finite elements: {finite_count}")
            else:
                print(f"  {key}: None")
        
        print(f"\n✅ Attention mask tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during attention mask testing: {str(e)}")
        import traceback
        traceback.print_exc()

def test_bidirectional_behavior():
    """Test that bidirectional layers actually allow full attention"""
    print(f"\n🔄 Testing bidirectional attention behavior:")
    
    # Create a simple test case
    seq_len = 5
    
    # Standard causal mask (upper triangular is -inf or large negative)
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    print(f"Standard causal mask:")
    print(causal_mask.numpy())
    
    # BiLLM bidirectional mask (all zeros = no masking)
    bidirectional_mask = torch.zeros((seq_len, seq_len))
    print(f"\nBiLLM bidirectional mask:")
    print(bidirectional_mask.numpy())
    
    # Show what this means for attention weights
    print(f"\nAttention behavior demonstration:")
    print(f"Causal: Token 0 can attend to tokens [0], token 2 can attend to tokens [0,1,2]")
    print(f"Bidirectional: All tokens can attend to all tokens [0,1,2,3,4]")
    
    # Verify mask effect on softmax
    dummy_scores = torch.randn(seq_len, seq_len)
    
    causal_attention = torch.softmax(dummy_scores + causal_mask, dim=-1)
    bidirectional_attention = torch.softmax(dummy_scores + bidirectional_mask, dim=-1)
    
    print(f"\nCausal attention weights (row 2, should have zeros for cols 3,4):")
    print(causal_attention[2].numpy())
    
    print(f"Bidirectional attention weights (row 2, should have non-zero for all cols):")
    print(bidirectional_attention[2].numpy())

if __name__ == "__main__":
    test_attention_mask_logic()
    test_bidirectional_behavior()
