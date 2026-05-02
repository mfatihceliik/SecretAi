import sys
import argparse
from src.utils.config_manager import config_manager

def main():
    parser = argparse.ArgumentParser(description="SecretAi Control Center")
    parser.add_argument("--mode", type=str, choices=["index", "train", "chat"], default="chat",
                        help="Mode: index (RAG), train (LLM Fine-tune), chat (Interact)")

    args = parser.parse_args()

    # --- LAZY LOADING MODULES ---
    if args.mode == "index":
        from src.core.rag_engine import RAGEngine
        print("[INFO] Starting RAG indexing (CPU/GPU compatible)...")
        engine = RAGEngine()
        engine.index_knowledge_base()
    
    elif args.mode == "train":
        try:
            from src.training.trainer import main as run_train
            print("[INFO] Starting Model Training (GPU REQUIRED)...")
            run_train()
        except NotImplementedError:
            print("[ERROR] ERROR: Unsloth could not find a GPU. NVIDIA GPU and CUDA are required for training.")
        except Exception as e:
            print(f"[ERROR] Training error: {e}")
        
    elif args.mode == "chat":
        print("[INFO] Starting Assistant...")
        try:
            from src.core.assistant import SecretAssistant
            bot = SecretAssistant()
            while True:
                q = input("\n[USER] You: ")
                if q.lower() in ["exit", "quit", "exit"]: break
                print(f"\n[AI] SecretAi:\n{bot.generate_response(q)}")
        except Exception as e:
            print(f"[ERROR] Error: {e}")
            print("Tip: If you don't have a GPU, change the model path in 'config.yaml' to the base model or complete training on a GPU machine.")

if __name__ == "__main__":
    main()
