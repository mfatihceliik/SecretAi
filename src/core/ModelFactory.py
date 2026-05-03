try:
    from unsloth import FastLanguageModel
except ImportError:
    FastLanguageModel = None

from src.utils.ConfigManager import config_manager


class ModelFactory:
    """
    ModelFactory handles the initialization of models and tokenizers,
    applying LoRA configurations driven by global settings.
    """
    @staticmethod
    def create_model_and_tokenizer():
        if FastLanguageModel is None:
            raise ImportError("Unsloth library not found. Cannot create model.")

        model_name = config_manager.get("training.base_model", "unsloth/Llama-3.2-1B-bnb-4bit")
        max_seq_length = config_manager.get("training.max_seq_length", 2048)
        lora_r = config_manager.get("training.lora_r", 32)
        lora_alpha = config_manager.get("training.lora_alpha", 32)
        
        # Target modules from config
        target_modules = config_manager.get("training.target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ])

        print(f"[INFO] Loading model: {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        
        print("[INFO] Applying LoRA adapters from configuration...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        
        return model, tokenizer
