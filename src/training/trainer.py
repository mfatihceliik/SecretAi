from src.core.model_factory import ModelFactory
from src.core.dataset_loader import SecretAiDatasetLoader
from src.utils.config_manager import config_manager
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import torch
import os
from datasets import load_dataset

class SecretAiTrainer:
    """
    SecretAiTrainer encapsulates the fine-tuning logic for the LLM.
    It manages model loading, dataset preparation, and the training loop.
    """
    def __init__(self):
        self.config = config_manager
        self.output_dir = self.config.get("paths.output_models", "outputs")

    def _load_train_dataset(self, tokenizer):
        dataset_path = self.config.get("paths.refined_kb", "data/secret_ai_final_massive_dataset.jsonl")
        
        if os.path.exists(dataset_path):
            print(f"[INFO] Loading pre-processed dataset from {dataset_path}...")
            return load_dataset("json", data_files=dataset_path, split="train")
        else:
            print("[INFO] Local dataset not found. Fetching from Hugging Face...")
            loader = SecretAiDatasetLoader(tokenizer)
            # Default to a safe limit if local file is missing
            return loader.load_and_prepare(max_samples=2000)

    def train(self):
        """
        Executes the full training pipeline.
        """
        # 1. Initialize Model & Tokenizer
        model, tokenizer = ModelFactory.create_model_and_tokenizer()

        # 2. Prepare Dataset
        train_dataset = self._load_train_dataset(tokenizer)

        # 3. Configure Trainer
        training_args = TrainingArguments(
            per_device_train_batch_size = self.config.get("training.batch_size", 2),
            gradient_accumulation_steps = self.config.get("training.grad_accum_steps", 4),
            warmup_steps = 5,
            max_steps = self.config.get("training.max_steps", 60),
            learning_rate = self.config.get("training.learning_rate", 2e-4),
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = self.output_dir,
            save_strategy = "no", # We save manually at the end
        )

        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = train_dataset,
            dataset_text_field = "text",
            max_seq_length = self.config.get("training.max_seq_length", 2048),
            dataset_num_proc = 2,
            packing = False,
            args = training_args,
        )

        # 4. Execute Training
        print("[INFO] Starting training...")
        trainer_stats = trainer.train()
        
        # 5. Save Model and Tokenizer
        print(f"[INFO] Saving model to {self.output_dir}...")
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        
        print(f"[SUCCESS] Training completed in {trainer_stats.metrics['train_runtime']:.2f} seconds.")
        print(f"[INFO] Weights saved to: {os.path.abspath(self.output_dir)}")

def main():
    trainer = SecretAiTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
