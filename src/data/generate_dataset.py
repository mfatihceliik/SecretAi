import argparse
from src.core.dataset_loader import SecretAiDatasetLoader

def main():
    parser = argparse.ArgumentParser(description="SecretAi Data Harvester")
    parser.add_argument("--per_lang", type=int, default=20000, help="Samples per language for Stack v2")
    parser.add_argument("--type", type=str, default="stack", choices=["stack", "magic", "both"], 
                        help="Which dataset to harvest")
    args = parser.parse_args()

    loader = SecretAiDatasetLoader()

    if args.type in ["magic", "both"]:
        print("\n--- Harvesting MAGICODER ---")
        magic_dataset = loader.harvest_magicoder()
        if magic_dataset:
            loader.save_dataset(magic_dataset, "data/magicoder_logic.jsonl")

    if args.type in ["stack", "both"]:
        # The loader now handles individual language saves internally
        loader.harvest_stack_v2(samples_per_lang=args.per_lang)

    print("\n[FINISHED]")
    print("Files are saved in 'data/' folder individually by language.")

if __name__ == "__main__":
    main()
