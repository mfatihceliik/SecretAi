import json
import re
import os
from src.utils.config_manager import config_manager
from src.core.models import KnowledgeChunk

class KnowledgeProcessor:
    """
    KnowledgeProcessor handles cleaning and chunking of raw harvested documentation
    to prepare it for RAG indexing, driven by configuration.
    """
    def __init__(self):
        self.config = config_manager
        self.primary_cat = self.config.get('categories.primary', 'Backend')
        self.secondary_cat = self.config.get('categories.secondary', 'Android')
        self.noise_patterns = self.config.get('processing.noise_patterns', [])
        self.min_length = self.config.get('processing.min_content_length', 150)

    def _clean_noise(self, text: str) -> str:
        for pattern in self.noise_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
        
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _chunk_text(self, text: str, limit=500) -> list:
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

    def refine_dataset(self):
        """
        Processes raw harvested data into refined knowledge chunks using config-driven logic.
        """
        input_file = self.config.get('paths.raw_data', 'data/knowledge_base_massive.json')
        output_file = self.config.get('paths.refined_kb', 'data/knowledge_base_refined.json')
        
        if not os.path.exists(input_file):
            print(f"[ERROR] {input_file} not found!")
            return

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        refined_data = []
        
        for doc in data:
            raw_content = doc.get("content", "")
            clean_content = self._clean_noise(raw_content)
            
            if len(clean_content) < self.min_length:
                continue
                
            chunks = self._chunk_text(clean_content, limit=600)
            
            category = doc["category"]
            # Migration logic mapped to config categories
            if category == "Kotlin":
                category = self.secondary_cat
            
            for chunk in chunks:
                code_snippet = ""
                if doc.get("code_snippets"):
                    code_snippet = doc["code_snippets"][0]
                
                knowledge_chunk = KnowledgeChunk(
                    category=category,
                    topic=doc["topic"],
                    content=chunk,
                    code=code_snippet,
                    source=doc["source"]
                )
                refined_data.append(knowledge_chunk.to_dict())

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(refined_data, f, ensure_ascii=False, indent=2)
        
        print(f"[SUCCESS] {len(refined_data)} chunks processed into {output_file}.")

if __name__ == "__main__":
    processor = KnowledgeProcessor()
    processor.refine_dataset()
