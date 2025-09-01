import os
# Set the environment variable to enable bidirectional mode
os.environ['BiLLM_START_INDEX'] = '0'

# Import your converted model (replace with your specific model type)
from src.billm import LlamaForSequenceClassification, Gemma3ForSequenceClassification

# Load converted model
# model = LlamaForSequenceClassification.from_pretrained("models/billm_llama_31_8b_laerebogen")
model = Gemma3ForSequenceClassification.from_pretrained("models/billm_gemma_3_1b_cpt_dynaword_full_v1")

# Quick test: Check if bidirectional layers exist
assert model is not None
assert len(model.model.bidirectionas) > 0

print(f"✅ Conversion successful! Found {len(model.model.bidirectionas)} bidirectional layers")