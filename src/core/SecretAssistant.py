from unsloth import FastLanguageModel
from src.core.RAGEngine import RAGEngine
from src.utils.ConfigManager import config_manager

class SecretAssistant:
    """
    SecretAssistant provides a unified interface for interacting with the fine-tuned LLM,
    enriched with RAG capabilities and fully driven by configuration templates.
    """
    def __init__(self):
        self.rag = RAGEngine()
        self.config = config_manager
        
        if FastLanguageModel is None:
            raise ImportError("Unsloth library not found. Please install it to use SecretAssistant.")

        print("🚀 Loading model for inference...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.config.get("paths.output_models", "models/assistant"),
            max_seq_length = self.config.get("training.max_seq_length", 2048),
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(self.model)

    def generate_response(self, query: str, category: str = None) -> str:
        """
        Generates a context-aware response using Hybrid RAG and the fine-tuned model.
        """
        # 1. Hybrid RAG Search
        docs = self.rag.search(query, category)
        
        if docs:
            # Format context with numbered entries for better model attention
            context = "\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(docs)])
        else:
            context = "No specific project documentation found. Use general software engineering best practices."
        
        # 2. System prompt and Template from config
        system_prompt = self.config.get("prompts.assistant.system")
        prompt_tpl = self.config.get("prompts.assistant.template")

        # 3. Construct Final Prompt
        prompt = prompt_tpl.format(
            system_prompt=system_prompt,
            context=context,
            query=query
        )
        
        # 4. Tokenization and Inference
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=self.config.get("assistant.max_new_tokens", 512),
            temperature=self.config.get("assistant.temperature", 0.5), # Lower temp for more factual coding
            do_sample=True,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 5. Extraction
        if "### Response:" in full_text:
            return full_text.split("### Response:")[1].strip()
        
        # Fallback extraction if template was slightly ignored
        return full_text.strip()

if __name__ == "__main__":
    try:
        assistant = SecretAssistant()
        print("🤖 SecretAssistant ready!")
    except Exception as e:
        print(f"[ERROR] Could not initialize assistant: {e}")
