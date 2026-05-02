from src.core.rag_engine import RAGEngine
from src.utils.config_manager import config_manager

class SecretAssistant:
    def __init__(self):
        self.rag = RAGEngine()
        
        print("🚀 Loading model...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = config_manager.get("paths.output_models", "outputs"),
            max_seq_length = config_manager.get("training.max_seq_length", 2048),
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(self.model)

    def generate_response(self, query, category=None):
        # RAG Search
        docs = self.rag.search(query, category)
        context = "\n".join(docs) if docs else "General best practices."
        
        prompt = f"""### System:\nYou are a senior engineer. Use the context below.\n\n### Context:\n{context}\n\n### Instruction:\n{query}\n\n### Response:\n"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=config_manager.get("assistant.max_new_tokens", 500),
            temperature=config_manager.get("assistant.temperature", 0.7),
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_text.split("### Response:")[1].strip()

if __name__ == "__main__":
    assistant = SecretAssistant()
    print("🤖 Assistant ready! Ask your question...")
