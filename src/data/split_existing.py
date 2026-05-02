import json
import os

def split_existing_dataset(input_file="data/stack_raw_code.jsonl"):
    if not os.path.exists(input_file):
        print(f"[!] Dosya bulunamadı: {input_file}")
        return

    print(f"Bölme işlemi başlıyor: {input_file}")
    
    # Dosyaları açalım
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

    print("\n--- İşlem Tamamlandı ---")
    print(f"Kotlin: {counts['Kotlin']} örnek ayrıldı.")
    print(f"Python: {counts['Python']} örnek ayrıldı.")
    if counts["Other"] > 0:
        print(f"Bilinmeyen/Diğer: {counts['Other']} örnek.")
    
    print("\nArtık 'generate_dataset.py' komutunu çalıştırabilirsin. Sistem bu dilleri atlayacaktır.")

if __name__ == "__main__":
    split_existing_dataset()
