# load model to evaluate
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer
from billm import Gemma3ForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import evaluate
import torch

# load peft model
from peft import PeftConfig, PeftModel

envs_dir = os.getcwd() + '/envs.env'
print(f"Loading envs from: {envs_dir}")
load_dotenv(envs_dir)
hf_token = os.getenv("HF_TOKEN")
hf_token_euroeval = os.getenv("HF_TOKEN_EUROEVAL")
wandb_api_key = os.getenv("WANDB_API_KEY")

base_model_path = "../new_models/student_step31816_2dyna"
adapter_path = "billm_angry_tweets_new_models-student_step31816_2dyna_ckpt_1"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = Gemma3ForSequenceClassification.from_pretrained(base_model_path)

peft_config = PeftConfig.from_pretrained(adapter_path)

peft_model = PeftModel.from_pretrained(model, adapter_path)

# load dataset
from datasets import load_dataset

ds = load_dataset("EuroEval/angry-tweets-mini", token=hf_token_euroeval)
label2id = {"positive": 0, "neutral": 1, "negative": 2}




