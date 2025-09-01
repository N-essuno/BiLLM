import os
# Set the start index for the BiLLM package
os.environ['BiLLM_START_INDEX'] = '0' # All layers bi-directional
# os.environ['BiLLM_START_INDEX'] = '16' # Only layers 16+ bi-directional
# os.environ['BiLLM_START_INDEX'] = '-1' # Disable BiLLM (uni-directional)

from dotenv import load_dotenv, find_dotenv
load_dotenv(os.getcwd() + '/envs.env')
hf_token = os.getenv("HF_TOKEN")

# Instead of importing from transformers
# from src.billm import (
#     LlamaModel,
#     LlamaForCausalLM,
#     LlamaForSequenceClassification,
#     LlamaForTokenClassification
# )

from transformers import AutoTokenizer
from src.billm import LlamaForSequenceClassification, Gemma3ForSequenceClassification

# model_name_hf = "danish-foundation-models/Meta-Llama-3.1-8B-laerebogen"
model_name_hf = "danish-foundation-models/gemma-3-1b-cpt-dynaword-full-v1"

match model_name_hf:
    case "danish-foundation-models/Meta-Llama-3.1-8B-laerebogen":
        model = LlamaForSequenceClassification.from_pretrained(model_name_hf, token=hf_token)
        local_dir = "models/billm_llama_31_8b_laerebogen"
    case "danish-foundation-models/gemma-3-1b-cpt-dynaword-full-v1":
        model = Gemma3ForSequenceClassification.from_pretrained(model_name_hf, token=hf_token)
        local_dir = "models/billm_gemma_3_1b_cpt_dynaword_full_v1"
    case _:
        raise ValueError(f"Model {model_name_hf} not supported.")

model.save_pretrained(local_dir)

tokenizer = AutoTokenizer.from_pretrained(model_name_hf, token=hf_token)
tokenizer.save_pretrained(local_dir)

print(f"Model and tokenizer saved to {local_dir}")