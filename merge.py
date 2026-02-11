import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

# 1. Define paths
base_model_path = "Qwen/Qwen2.5-VL-3B-Instruct"  # Your Qwen2.5 model
adapter_path = "/data1/hbshim/Qwen3-VL/qwen-vl-finetune/output_lora"        # Your adapter path
save_path = "/data1/hbshim/Qwen3-VL/qwen-vl-finetune/merged_model"

print(f"Loading base model from {base_model_path}...")

# CHANGE: Use AutoModelForVision2Seq instead of specific class
base_model = AutoModelForVision2Seq.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,  
    device_map="auto",
    # trust_remote_code=True # Uncomment if using a very new version/custom model code
)

print(f"Loading adapter from {adapter_path}...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {save_path}...")
model.save_pretrained(save_path)

print("Saving processor and configuration...")
processor = AutoProcessor.from_pretrained(base_model_path)
processor.save_pretrained(save_path)

print(f"Done! Saved to {save_path}")