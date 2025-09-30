import torch
from dotenv import load_dotenv
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

load_dotenv()

student = "google/gemma-3-1b-pt"
attn_implementation = "eager"
max_seq_length = 4096

# pt_model_path = "../../training/offpolicy_kd/checkpoints/cosine_dyna_gemma3_1b_full/student_step12736.pt"
dest_root = "dyna_vs_giga/"
dest_root_2 = "../new_models/"
base_path = "../../training/offpolicy_kd/checkpoints/"
base_path_2 = "../new_pt_models/"

dyna_vs_giga_base_path = "../../training/offpolicy_kd/checkpoints/"
dyna_pt_paths = [
    base_path + "cosine_dyna_gemma3_1b_full/student_step12736.pt",
    # base_path + "cosine_dyna_gemma3_1b_distill_full/student_step12736.pt",
    base_path + "cosine_dyna_gemma3_1b_7356/student_step7356.pt",
    # base_path + "cosine_dyna_gemma3_1b_distill_7356/student_step7356.pt",
    base_path + "cosine_dyna_gemma3_1b_full_fromscratch/student_step12736.pt",
    # base_path + "cosine_dyna_gemma3_1b_distill_full_fromscratch/student_step12736.pt",
    base_path + "cosine_dyna_gemma3_1b_7356_fromscratch/student_step7356.pt",
    # base_path + "cosine_dyna_gemma3_1b_distill_7356_fromscratch/student_step7356.pt"
]

giga_pt_paths = [
    base_path + "cosine_giga_gemma3_1b_full/student_step7356.pt",
    # base_path + "cosine_giga_gemma3_1b_distill_full/student_step7356.pt",
    base_path + "cosine_giga_gemma3_1b_full_fromscratch/student_step7356.pt",
    # base_path + "cosine_giga_gemma3_1b_distill_full_fromscratch/student_step7356.pt"
]

# new_pt_paths = [
#     base_path_2 + "student_step15908_dyna_4b.pt",
#     base_path_2 + "student_step15908_dyna_none.pt",
#     base_path_2 + "student_step31816_2dyna.pt",
# ]

new_pt_paths = [
    # base_path_2 + "student_step3678_onpolicy.pt",
    # base_path_2 + "student_step7356_cos_giga_distill_full.pt",
    base_path_2 + "student_step24125_dyna_commonpile.pt",
]

print(f"Loading model from {student}")
student_config = AutoConfig.from_pretrained(student)
student_config.attn_implementation = attn_implementation
student_config.max_position_embeddings = max_seq_length
student_model = AutoModelForCausalLM.from_pretrained(student, config=student_config, attn_implementation='eager')

print(f"Loading tokenizer from {student}")
tokenizer = AutoTokenizer.from_pretrained(student)

for pt_model_path in new_pt_paths:
    print(f"Loading checkpoint from {pt_model_path}")

    if pt_model_path is not None:
        state_dict = torch.load(pt_model_path, map_location="cpu")
        # Remove common prefixes that can appear in saved checkpoints
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        student_model.load_state_dict(state_dict)
    else:
        raise ValueError("No checkpoint provided to load the model from.")

    # Take the last file name (-2 for folder name) and concatenate it to the root destination directory
    dest_path = dest_root_2 + pt_model_path.split("/")[-1] + "/"
    # remove .pt suffix
    dest_path = dest_path.removesuffix(".pt/") # if selecting filename instead of folder name
    print(f"Saving model to {dest_path}")
    student_model.save_pretrained(dest_path)
    tokenizer.save_pretrained(dest_path)

    

