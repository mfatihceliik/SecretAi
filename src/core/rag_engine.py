import json
import os
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from src.utils.config_manager import config_manager
from src.core.models import KnowledgeChunk

class RAGEngine:
    """
    RAGEngine manages the vector database (ChromaDB) and provides search capabilities
    using Sentence Transformers embeddings, driven by global configuration.
    """
    def __init__(self):
        self.config = config_manager
        # Check for GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Embedding device: {self.device.upper()}")
        
        self.embed_model = SentenceTransformer(
            self.config.get('rag.embedding_model', 'all-MiniLM-L6-v2'), 
            device=self.device
        )
        self.client = chromadb.PersistentClient(path=self.config.get('paths.chroma_db', 'chroma_db'))
        self.collection = self.client.get_or_create_collection("docs")

    def chunk_text(self, text: str):
        chunk_size = self.config.get('rag.chunk_size', 500)
        overlap = self.config.get('rag.chunk_overlap', 50)
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
        return chunks

    def index_knowledge_base(self):
        """
        Processes and indexes the refined knowledge base into ChromaDB.
        """
        if self.collection.count() > 0:
            print(f"[INFO] Database already contains {self.collection.count()} records. Skipping indexing.")
            return

        json_file = self.config.get('paths.refined_kb', 'data/refined_kb.json')
        if not os.path.exists(json_file):
            print(f"[ERROR] Knowledge base file not found: {json_file}")
            return

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"[INFO] Chunking and preparing {len(data)} documents...")
        
        all_texts = []
        all_metadatas = []
        all_ids = []

        for i, doc in tqdm(enumerate(data), total=len(data), desc="Chunking"):
            content = doc.get("content", "")
            chunks = self.chunk_text(content)
            for j, chunk in enumerate(chunks):
                knowledge_chunk = KnowledgeChunk(
                    category=doc.get("category", "General"),
                    topic=doc.get("topic", "N/A"),
                    content=chunk,
                    code=doc.get("code", ""),
                    source=doc.get("source", "")
                )
                
                all_texts.append(knowledge_chunk.content)
                all_metadatas.append({
                    "category": knowledge_chunk.category,
                    "topic": knowledge_chunk.topic,
                    "source": knowledge_chunk.source
                })
                all_ids.append(f"id_{i}_{j}")

        print(f"[INFO] Vectorizing and adding {len(all_texts)} chunks to DB...")
        
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
        
        print(f"[SUCCESS] Indexing completed! Total: {self.collection.count()} records.")

    def search(self, query: str, category: str = None):
        """
        Performs a vector search in ChromaDB using parameters from configuration.
        """
        emb = self.embed_model.encode(query).tolist()
        where = {"category": category} if category else None
        results = self.collection.query(
            query_embeddings=[emb],
            n_results=self.config.get('rag.n_results', 7),
            where=where
        )
        return results["documents"][0] if results["documents"] else []

if __name__ == "__main__":
    engine = RAGEngine()
    engine.index_knowledge_base()
