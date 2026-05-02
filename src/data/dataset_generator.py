import argparse
from src.core.secret_ai_dataset_loader import SecretAiDatasetLoader

class DatasetGenerator:
    """
    DatasetGenerator orchestrates the full dataset harvesting pipeline.
    It combines instruction-based logic and raw code harvesting.
    """
    def __init__(self, per_lang_limit=20000, magicoder_limit=110000):
        self.loader = SecretAiDatasetLoader()
        self.per_lang_limit = per_lang_limit
        self.magicoder_limit = magicoder_limit

    def generate(self, mode="both"):
        """
        Runs the generation process based on the specified mode.
        """
        print(f"--- SecretAi Dataset Generation Starting (Mode: {mode}) ---")
        
        if mode in ["magicoder", "both"]:
            magic_dataset = self.loader.harvest_magicoder(limit=self.magicoder_limit)
            if magic_dataset:
                self.loader.save_dataset(magic_dataset, "data/magicoder_logic.jsonl")

        if mode in ["stack", "both"]:
            # The loader handles individual language saves internally
            self.loader.harvest_stack_v2(samples_per_lang=self.per_lang_limit)

        print("\n[FINISHED] Dataset generation completed.")

def main():
    parser = argparse.ArgumentParser(description="SecretAi Dataset Generator CLI")
    parser.add_argument("--type", type=str, choices=["magicoder", "stack", "both"], default="both")
    parser.add_argument("--per-lang", type=int, default=20000)
    args = parser.parse_args()

    generator = DatasetGenerator(per_lang_limit=args.per_lang)
    generator.generate(mode=args.type)

if __name__ == "__main__":
    main()
