import glob
import os

def merge_jsonl(output_filename="data/secret_ai_final_massive_dataset.jsonl"):
    # data/stack_*.jsonl ve magicoder_logic.jsonl dosyalarını bul
    files = glob.glob("data/stack_*.jsonl")
    if os.path.exists("data/magicoder_logic.jsonl"):
        files.append("data/magicoder_logic.jsonl")
        
    print(f"Combining {len(files)} files...")
    
    with open(output_filename, "w", encoding="utf-8") as outfile:
        for fname in files:
            print(f"Adding {fname}...")
            with open(fname, "r", encoding="utf-8") as infile:
                for line in infile:
                    outfile.write(line)
                    
    print(f"\n[SUCCESS] Final dataset ready: {output_filename}")

if __name__ == "__main__":
    merge_jsonl()
