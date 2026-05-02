from unsloth import FastLanguageModel
from src.utils.config_manager import config_manager
import torch

class ModelFactory:
    @staticmethod
    def create_model_and_tokenizer():
        model_name = config_manager.get("training.base_model", "unsloth/Llama-3.2-1B-bnb-4bit")
        max_seq_length = config_manager.get("training.max_seq_length", 2048)
        lora_r = config_manager.get("training.lora_r", 32)
        lora_alpha = config_manager.get("training.lora_alpha", 32)

        print(f"[INFO] Loading model: {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        
        print("[INFO] Applying LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        
        return model, tokenizer
