import json
import os
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from src.utils.ConfigManager import config_manager
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
        self.client = chromadb.PersistentClient(path=self.config.get('paths.rag_chroma_db', 'data/rag/chroma_db'))
        self.collection = self.client.get_or_create_collection("docs")

    def chunk_text(self, text: str):
        """
        Smart recursive-style chunking that respects line breaks and keeps context together.
        """
        chunk_size = self.config.get('rag.chunk_size', 500)
        overlap = self.config.get('rag.chunk_overlap', 50)
        
        # If it's a small text, return as is
        if len(text.split()) <= chunk_size:
            return [text]

        # Split by paragraphs or double newlines first
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk.split()) + len(para.split()) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If paragraph itself is too big, split by sentences
                if len(para.split()) > chunk_size:
                    sentences = para.replace(".", ".\n").split("\n")
                    sub_chunk = ""
                    for sent in sentences:
                        if len(sub_chunk.split()) + len(sent.split()) <= chunk_size:
                            sub_chunk += sent + " "
                        else:
                            chunks.append(sub_chunk.strip())
                            sub_chunk = sent + " "
                    current_chunk = sub_chunk
                else:
                    current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def index_knowledge_base(self):
        """
        Processes and indexes the refined knowledge base into ChromaDB.
        """
        # We allow re-indexing if needed or appending
        current_count = self.collection.count()
        print(f"[INFO] Database current record count: {current_count}")

        # Point to the final merged knowledge base
        json_file = self.config.get('paths.rag_final_kb', 'data/rag/final/final_kb.json')
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
        Performs a Hybrid Search combining Vector Search and Keyword Search.
        """
        # 1. Vector Search (Semantic)
        emb = self.embed_model.encode(query).tolist()
        where = {"category": category} if category else None
        
        vector_results = self.collection.query(
            query_embeddings=[emb],
            n_results=self.config.get('rag.n_results', 5),
            where=where
        )
        
        # 2. Keyword Search (Literal) - Simplified Hybrid approach
        # We extract key terms from query for a secondary literal search
        keywords = query.split()[:3] # Take first 3 words as keywords
        keyword_results = []
        
        if keywords:
            # We use ChromaDB's where_document filter for a primitive keyword search
            # Note: ChromaDB supports $contains for metadata/document
            kw_query = {"$or": [{"$contains": kw} for kw in keywords]} if len(keywords) > 1 else {"$contains": keywords[0]}
            
            kw_search_res = self.collection.query(
                query_embeddings=[emb], # We still need embeddings but results are filtered by keywords
                n_results=self.config.get('rag.n_results', 3),
                where_document=kw_query,
                where=where
            )
            keyword_results = kw_search_res["documents"][0] if kw_search_res["documents"] else []

        # Merge results while maintaining order (Vector first, then Keyword additions)
        all_docs = vector_results["documents"][0] if vector_results["documents"] else []
        
        for doc in keyword_results:
            if doc not in all_docs:
                all_docs.append(doc)
        
        return all_docs[:self.config.get('rag.n_results', 7)]

if __name__ == "__main__":
    engine = RAGEngine()
    engine.index_knowledge_base()
