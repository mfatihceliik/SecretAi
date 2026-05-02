import json
import re
import os
from src.utils.config_manager import config_manager

def clean_noise(text):
    noise_patterns = [
        r"Stay organized with collections.*?\.",
        r"Save and categorize content based on your preferences\.",
        r"Added in [\d\.]+",
        r"Artifact: [\w\.\:]+",
        r"View Source",
        r"Kotlin \| Java",
        r"Public functions",
        r"Parameters <T>.*?\n",
        r"Throws .*?\n"
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, limit=500):
    sentences = re.split(r'(?<=[.\?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < limit:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def refine_dataset():
    input_file = config_manager.get('paths.raw_data', 'data/knowledge_base_massive.json')
    output_file = config_manager.get('paths.refined_kb', 'data/knowledge_base_refined.json')
    
    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found!")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    refined_data = []
    
    for doc in data:
        raw_content = doc.get("content", "")
        clean_content = clean_noise(raw_content)
        
        if len(clean_content) < 150:
            continue
            
        chunks = chunk_text(clean_content, limit=600)
        
        # --- KATEGORİ DÜZELTME ---
        # Eğer kategori 'Kotlin' ise bunu 'Android'e taşıyoruz (Yapıyı bozmamak için)
        category = doc["category"]
        if category == "Kotlin":
            category = "Android"
        
        for i, chunk in enumerate(chunks):
            code_snippet = ""
            if doc.get("code_snippets"):
                code_snippet = doc["code_snippets"][0]
            
            refined_data.append({
                "category": category, # Artık sadece Android veya Backend
                "topic": doc["topic"],
                "content": chunk,
                "code": code_snippet,
                "source": doc["source"]
            })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(refined_data, f, ensure_ascii=False, indent=2)
    
    print(f"SUCCESS: {len(refined_data)} chunks mapped to Android and Backend.")
    print("Kotlin category successfully merged into Android Pillar.")

if __name__ == "__main__":
    refine_dataset()
