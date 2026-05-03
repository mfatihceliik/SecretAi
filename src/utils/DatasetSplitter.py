import json
import os
from src.utils.ConfigManager import config_manager

class DatasetSplitter:
    """
    DatasetSplitter provides utility to separate a mixed dataset file
    into language-specific JSONL files based on content markers.
    """
    @staticmethod
    def split_existing_dataset(input_file=None):
        datasets_dir = config_manager.get('paths.training_datasets_dir', 'data/training/datasets')
        
        if input_file is None:
            # Default fallback if not provided
            input_file = os.path.join(datasets_dir, "stack_raw_code.jsonl")

        if not os.path.exists(input_file):
            print(f"[ERROR] File not found: {input_file}")
            return

        print(f"[INFO] Starting split process for: {input_file}")
        
        # Ensure training datasets directory exists
        os.makedirs(datasets_dir, exist_ok=True)
        
        # Open files for specific languages
        kotlin_path = os.path.join(datasets_dir, "stack_Kotlin.jsonl")
        python_path = os.path.join(datasets_dir, "stack_Python.jsonl")
        
        kotlin_file = open(kotlin_path, "w", encoding="utf-8")
        python_file = open(python_path, "w", encoding="utf-8")
        
        counts = {"Kotlin": 0, "Python": 0, "Other": 0}

        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    
                    if "### Language: Kotlin" in text:
                        kotlin_file.write(line)
                        counts["Kotlin"] += 1
                    elif "### Language: Python" in text:
                        python_file.write(line)
                        counts["Python"] += 1
                    else:
                        counts["Other"] += 1
                except Exception:
                    continue

        kotlin_file.close()
        python_file.close()

        print("\n--- Split Process Completed ---")
        print(f"  - Kotlin: {counts['Kotlin']} samples separated -> {kotlin_path}")
        print(f"  - Python: {counts['Python']} samples separated -> {python_path}")
        if counts["Other"] > 0:
            print(f"  - Unknown/Other: {counts['Other']} samples skipped.")
        
        print("\n[INFO] You can now run the dataset generator. Existing language files will be skipped.")

if __name__ == "__main__":
    DatasetSplitter.split_existing_dataset()
