import glob
import os
from src.utils.ConfigManager import config_manager

class DatasetMerger:
    """
    DatasetMerger provides utility to combine multiple JSONL dataset files
    into a single master dataset for training.
    """
    @staticmethod
    def merge_jsonl():
        datasets_dir = config_manager.get('paths.training_datasets_dir', 'data/training/datasets')
        output_filename = config_manager.get('paths.training_final_dataset', 'data/training/final_dataset.jsonl')
        
        # Find stack_*.jsonl and magicoder_logic.jsonl files in the training dir
        files = glob.glob(os.path.join(datasets_dir, "stack_*.jsonl"))
        magicoder_path = os.path.join(datasets_dir, "magicoder_logic.jsonl")
        
        if os.path.exists(magicoder_path):
            files.append(magicoder_path)
            
        if not files:
            print(f"[WARNING] No dataset files found in {datasets_dir}")
            return

        print(f"[INFO] Combining {len(files)} files into {output_filename}...")
        
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        
        with open(output_filename, "w", encoding="utf-8") as outfile:
            for fname in files:
                print(f"  -> Adding {fname}...")
                with open(fname, "r", encoding="utf-8") as infile:
                    for line in infile:
                        outfile.write(line)
                        
        print(f"\n[SUCCESS] Final dataset ready: {output_filename}")

if __name__ == "__main__":
    DatasetMerger.merge_jsonl()
