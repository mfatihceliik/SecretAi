import sys
import argparse
from src.utils.config_manager import config_manager

def main():
    parser = argparse.ArgumentParser(description="SecretAi Control Center")
    parser.add_argument("--mode", type=str, choices=["index", "train", "chat"], default="chat",
                        help="Mode: index (RAG), train (LLM Fine-tune), chat (Interact)")

    args = parser.parse_args()

    # --- MODÜLLERİ SADECE İHTİYAÇ ANINDA YÜKLE (Lazy Loading) ---
    if args.mode == "index":
        from src.core.rag_engine import RAGEngine
        print("[INFO] RAG Indeksleme başlatılıyor (CPU/GPU uyumlu)...")
        engine = RAGEngine()
        engine.index_knowledge_base()
    
    elif args.mode == "train":
        try:
            from src.training.trainer import main as run_train
            print("[INFO] Model Eğitimi başlatılıyor (GPU GEREKLİ)...")
            run_train()
        except NotImplementedError:
            print("[ERROR] HATA: Unsloth bir GPU bulamadı. Eğitim için NVIDIA GPU ve CUDA gereklidir.")
        except Exception as e:
            print(f"[ERROR] Eğitim hatası: {e}")
        
    elif args.mode == "chat":
        print("[INFO] Asistan Başlatılıyor...")
        try:
            from src.core.assistant import SecretAssistant
            bot = SecretAssistant()
            while True:
                q = input("\n[USER] Sen: ")
                if q.lower() in ["exit", "quit", "çık"]: break
                print(f"\n[AI] SecretAi:\n{bot.generate_response(q)}")
        except Exception as e:
            print(f"[ERROR] Hata: {e}")
            print("İpucu: Eğer GPU yoksa 'config.yaml' içindeki model yolunu ana modelle değiştirin veya eğitimi GPU'lu bir makinede tamamlayın.")

if __name__ == "__main__":
    main()
