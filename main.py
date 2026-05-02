import sys
import argparse
from src.utils.config_manager import config_manager

class SecretAiApp:
    """
    Main Application class for SecretAi.
    Orchestrates different modes of operation: indexing, training, and chatting.
    """
    def __init__(self):
        self.parser = self._setup_args()
        self.config = config_manager

    def _setup_args(self):
        parser = argparse.ArgumentParser(description="SecretAi Control Center")
        parser.add_argument(
            "--mode", 
            type=str, 
            choices=["index", "train", "chat", "harvest"], 
            default="chat",
            help="Operation mode: index (RAG), train (LLM Fine-tune), chat (Interact), harvest (Scrape Docs)"
        )
        return parser

    def run_index(self):
        from src.core.rag_engine import RAGEngine
        print("[INFO] Starting RAG indexing (CPU/GPU compatible)...")
        engine = RAGEngine()
        engine.index_knowledge_base()

    def run_train(self):
        try:
            from src.training.trainer import main as run_train
            print("[INFO] Starting Model Training (GPU REQUIRED)...")
            run_train()
        except NotImplementedError:
            print("[ERROR] Unsloth could not find a GPU. NVIDIA GPU and CUDA are required for training.")
        except Exception as e:
            print(f"[ERROR] Training error: {e}")

    def run_chat(self):
        print("[INFO] Starting Assistant...")
        try:
            from src.core.assistant import SecretAssistant
            bot = SecretAssistant()
            print("\n" + "="*50)
            print("SecretAi Assistant is ready. Type 'exit' to quit.")
            print("="*50)
            while True:
                q = input("\n[USER] You: ")
                if q.lower() in ["exit", "quit"]:
                    break
                
                response = bot.generate_response(q)
                print(f"\n[AI] SecretAi:\n{response}")
        except Exception as e:
            print(f"[ERROR] Chat error: {e}")
            print("Tip: Ensure the model path in 'config/config.yaml' points to your fine-tuned weights.")

    def run_harvest(self):
        from src.data.harvester import KnowledgeHarvester
        print("[INFO] Starting Knowledge Harvesting...")
        harvester = KnowledgeHarvester()
        harvester.run()

    def start(self):
        args = self.parser.parse_args()
        
        if args.mode == "index":
            self.run_index()
        elif args.mode == "train":
            self.run_train()
        elif args.mode == "chat":
            self.run_chat()
        elif args.mode == "harvest":
            self.run_harvest()

if __name__ == "__main__":
    app = SecretAiApp()
    app.start()
