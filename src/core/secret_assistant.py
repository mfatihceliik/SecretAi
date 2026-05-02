try:
    from unsloth import FastLanguageModel
except ImportError:
    FastLanguageModel = None

from src.core.rag_engine import RAGEngine
from src.utils.config_manager import config_manager
import torch

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
        Generates a context-aware response using RAG and the fine-tuned model,
        using prompt templates from config.yaml.
        """
        # RAG Search
        docs = self.rag.search(query, category)
        context = "\n".join(docs) if docs else "General software engineering best practices."
        
        # System prompt and Template from config
        system_prompt = self.config.get("prompts.assistant.system")
        prompt_tpl = self.config.get("prompts.assistant.template")

        prompt = prompt_tpl.format(
            system_prompt=system_prompt,
            context=context,
            query=query
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=self.config.get("assistant.max_new_tokens", 500),
            temperature=self.config.get("assistant.temperature", 0.7),
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the response part
        if "### Response:" in full_text:
            return full_text.split("### Response:")[1].strip()
        return full_text.strip()

if __name__ == "__main__":
    try:
        assistant = SecretAssistant()
        print("🤖 SecretAssistant ready!")
    except Exception as e:
        print(f"[ERROR] Could not initialize assistant: {e}")
