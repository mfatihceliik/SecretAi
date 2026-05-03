import argparse
from src.utils.ConfigManager import config_manager

class SecretAiApp:
    """
    Main Application class for SecretAi.
    Orchestrates different modes of operation: indexing, training, chatting, harvesting, and generation.
    """
    def __init__(self):
        self.parser = self._setup_args()
        self.config = config_manager

    def _setup_args(self):
        parser = argparse.ArgumentParser(description="SecretAi Control Center")
        parser.add_argument(
            "--mode", 
            type=str, 
            choices=["index", "train", "chat", "harvest", "process", "generate"], 
            default="chat",
            help="Operation mode: index (RAG), train (LLM Fine-tune), chat (Interact), harvest (Collect), process (Clean/Refine), generate (Synth Dataset)"
        )
        return parser

    def run_index(self):
        from src.core.RAGEngine import RAGEngine
        print("[INFO] Starting RAG indexing (CPU/GPU compatible)...")
        engine = RAGEngine()
        engine.index_knowledge_base()

    def run_process(self):
        from src.data.KnowledgeProcessor import KnowledgeProcessor
        print("[INFO] Starting Knowledge Processing (Raw -> Processed -> Final)...")
        processor = KnowledgeProcessor()
        processor.process_all_domains()

    def run_train(self):
        try:
            from src.training.SecretAiTrainer import SecretAiTrainer
            print("[INFO] Starting Model Training (GPU REQUIRED)...")
            trainer = SecretAiTrainer()
            trainer.train()
        except ImportError as e:
            print(f"[ERROR] Import error: {e}. Ensure all dependencies (Unsloth, etc.) are installed.")
        except NotImplementedError:
            print("[ERROR] Unsloth could not find a GPU. NVIDIA GPU and CUDA are required for training.")
        except Exception as e:
            print(f"[ERROR] Training error: {e}")

    def run_chat(self):
        print("[INFO] Starting Assistant...")
        try:
            from src.core.SecretAssistant import SecretAssistant
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
        from src.data.HarvesterOrchestrator import HarvesterOrchestrator
        print("[INFO] Starting Multi-Domain Knowledge Harvesting (HF & Docs)...")
        orchestrator = HarvesterOrchestrator()
        orchestrator.run_all()

    def run_generate(self):
        from src.data.DatasetGenerator import DatasetGenerator
        print("[INFO] Starting Dataset Generation (Stack v2 & Magicoder)...")
        generator = DatasetGenerator()
        generator.generate(mode="both")

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
        elif args.mode == "process":
            self.run_process()
        elif args.mode == "generate":
            self.run_generate()

if __name__ == "__main__":
    app = SecretAiApp()
    app.start()
