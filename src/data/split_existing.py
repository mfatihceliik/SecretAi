import json
import os

def split_existing_dataset(input_file="data/stack_raw_code.jsonl"):
    if not os.path.exists(input_file):
        print(f"[!] File not found: {input_file}")
        return

    print(f"Starting split process: {input_file}")
    
    # Open files
    kotlin_file = open("data/stack_Kotlin.jsonl", "w", encoding="utf-8")
    python_file = open("data/stack_Python.jsonl", "w", encoding="utf-8")
    
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

    print("\n--- Process Completed ---")
    print(f"Kotlin: {counts['Kotlin']} samples separated.")
    print(f"Python: {counts['Python']} samples separated.")
    if counts["Other"] > 0:
        print(f"Unknown/Other: {counts['Other']} samples.")
    
    print("\nYou can now run 'generate_dataset.py'. The system will skip these languages.")

if __name__ == "__main__":
    split_existing_dataset()
