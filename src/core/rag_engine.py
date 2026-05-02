import json
import os
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from src.utils.config_manager import config_manager

class RAGEngine:
    def __init__(self):
        # GPU var mı kontrol et
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Embedding cihazı: {self.device.upper()}")
        
        self.embed_model = SentenceTransformer(config_manager.get('rag.embedding_model', 'all-MiniLM-L6-v2'), device=self.device)
        self.client = chromadb.PersistentClient(path=config_manager.get('paths.chroma_db', 'chroma_db'))
        self.collection = self.client.get_or_create_collection("docs")

    def chunk_text(self, text):
        chunk_size = config_manager.get('rag.chunk_size', 500)
        overlap = config_manager.get('rag.chunk_overlap', 50)
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
        return chunks

    def index_knowledge_base(self):
        if self.collection.count() > 0:
            print(f"[INFO] Veritabanında zaten {self.collection.count()} kayıt var. İşlem atlanıyor.")
            return

        json_file = config_manager.get('paths.refined_kb', 'data/refined_kb.json')
        if not os.path.exists(json_file):
            print(f"[ERROR] Knowledge base file not found: {json_file}")
            return

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"[INFO] {len(data)} döküman parçalanıyor ve hazırlanıyor...")
        
        all_texts = []
        all_metadatas = []
        all_ids = []

        for i, doc in tqdm(enumerate(data), total=len(data), desc="Chunking"):
            content = doc.get("content", "")
            chunks = self.chunk_text(content)
            for j, chunk in enumerate(chunks):
                all_texts.append(chunk)
                all_metadatas.append({
                    "category": doc.get("category", ""),
                    "topic": doc.get("topic", ""),
                    "source": doc.get("source", "")
                })
                all_ids.append(f"id_{i}_{j}")

        print(f"[INFO] {len(all_texts)} parça vektörleniyor ve DB'ye ekleniyor...")
        
        batch_size = 128
        for i in tqdm(range(0, len(all_texts), batch_size), desc="Indexing"):
            end = i + batch_size
            batch_texts = all_texts[i:end]
            embeddings = self.embed_model.encode(batch_texts, show_progress_bar=False).tolist()
            
            self.collection.add(
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=all_metadatas[i:end],
                ids=all_ids[i:end]
            )
        
        print(f"[SUCCESS] İndeksleme tamamlandı! Toplam: {self.collection.count()} kayıt.")

    def search(self, query, category=None):
        emb = self.embed_model.encode(query).tolist()
        where = {"category": category} if category else None
        results = self.collection.query(
            query_embeddings=[emb],
            n_results=config_manager.get('rag.n_results', 7),
            where=where
        )
        return results["documents"][0] if results["documents"] else []

if __name__ == "__main__":
    engine = RAGEngine()
    engine.index_knowledge_base()
