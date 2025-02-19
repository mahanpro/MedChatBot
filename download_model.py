from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "meta-llama/Llama-3.2-3B"
access_token = "***"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=access_token, torch_dtype=torch.bfloat16)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Save model locally for later use
model.save_pretrained("./llama_3.2_3B")
tokenizer.save_pretrained("./llama_3.2_3B")
print("Model downloaded and saved locally.")
