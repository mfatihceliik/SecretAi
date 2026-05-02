from src.core.model_factory import ModelFactory
from src.core.dataset_loader import SecretAiDatasetLoader
from src.utils.config_manager import config_manager
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import torch
import os
from datasets import load_dataset

def main():
    # 1. Model & Tokenizer
    model, tokenizer = ModelFactory.create_model_and_tokenizer()

    # 2. Dataset Loading
    dataset_path = config_manager.get("paths.refined_kb", "data/secret_ai_final_massive_dataset.jsonl")
    
    if os.path.exists(dataset_path):
        print(f"[INFO] Loading pre-processed dataset from {dataset_path}...")
        train_dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        print("[INFO] Local dataset not found. Fetching from Hugging Face...")
        loader = SecretAiDatasetLoader(tokenizer)
        train_dataset = loader.load_and_prepare(max_samples=2000)

    # 3. Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text",
        max_seq_length = config_manager.get("training.max_seq_length", 2048),
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = config_manager.get("training.batch_size", 2),
            gradient_accumulation_steps = config_manager.get("training.grad_accum_steps", 4),
            warmup_steps = 5,
            max_steps = config_manager.get("training.max_steps", 60),
            learning_rate = config_manager.get("training.learning_rate", 2e-4),
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = config_manager.get("paths.output_models", "outputs"),
        ),
    )

    # 4. Train
    print("[INFO] Starting training...")
    trainer.train()
    
    # 5. Save
    save_path = config_manager.get("paths.output_models", "secret_ai_lora")
    print(f"[INFO] Saving model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("[SUCCESS] Training completed and model saved.")

if __name__ == "__main__":
    main()
