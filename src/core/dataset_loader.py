from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
from .dataset_processor import DatasetProcessor
from dotenv import load_dotenv
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from smart_open import open as smart_open
import concurrent.futures
from threading import Lock

load_dotenv()

class SecretAiDatasetLoader:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.processor = DatasetProcessor()
        self.hf_token = os.getenv("HF_TOKEN")
        # Increase pool capacity and add timeouts for connection stability
        self.s3_config = Config(
            signature_version=UNSIGNED, 
            max_pool_connections=20,
            connect_timeout=60,
            read_timeout=60
        )
        self.s3_client = boto3.client("s3", config=self.s3_config)
        self.lock = Lock()

    def _download_s3_content(self, blob_id, encoding="utf-8"):
        if not blob_id: return None
        s3_url = f"s3://softwareheritage/content/{blob_id}"
        try:
            with smart_open(s3_url, "rb", compression=".gz", transport_params={"client": self.s3_client}) as fin:
                return fin.read().decode(encoding or "utf-8", errors="ignore")
        except Exception: return None

    def process_stack_example(self, example, lang_name):
        """This function will now run in parallel as thread-safe"""
        code = self._download_s3_content(example.get("blob_id"), example.get("src_encoding"))
        if not code or len(str(code)) < 500: return None
        
        # Processor's filter sections are thread-safe
        example["content"] = code
        if not self.processor.is_good_code(example, lang_name): return None
            
        return self.processor.format_sample(code, lang_name)

    def harvest_stack_v2(self, samples_per_lang=20000):
        """Harvests each language into a separate file"""
        languages = self.processor.config_manager.languages
        print(f"--- SecretAi RESILIENT HARVESTER (Stack-v2) ---")
        
        max_workers = 10 

        for lang_cfg in languages:
            lang_name = lang_cfg["name"]
            output_path = f"data/stack_{lang_name}.jsonl"
            
            # Skip if this language already exists (to save time)
            if os.path.exists(output_path):
                print(f"\n[i] {lang_name} already exists, skipping: {output_path}")
                continue

            print(f"\n[🚀] Harvesting {lang_name} into {output_path}...")
            
            try:
                subset_stream = load_dataset("bigcode/the-stack-v2", name=lang_name, split="train", streaming=True, token=self.hf_token)
                lang_samples = []
                scanned_count = 0
                pbar = tqdm(total=samples_per_lang, desc=f"Collecting {lang_name}", unit=" file")

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = set()
                    stream_iter = iter(subset_stream)
                    
                    for _ in range(max_workers * 2):
                        try:
                            ex = next(stream_iter)
                            futures.add(executor.submit(self.process_stack_example, ex, lang_name))
                            scanned_count += 1
                        except StopIteration: break

                    while futures:
                        done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                        
                        for future in done:
                            processed = future.result()
                            if processed:
                                lang_samples.append(processed)
                                pbar.update(1)
                                pbar.set_postfix({"scanned": scanned_count})
                            
                            if len(lang_samples) < samples_per_lang and scanned_count < 2000000:
                                try:
                                    ex = next(stream_iter)
                                    futures.add(executor.submit(self.process_stack_example, ex, lang_name))
                                    scanned_count += 1
                                except StopIteration: pass

                pbar.close()
                
                # Save only this language
                if lang_samples:
                    self.save_dataset(Dataset.from_list(lang_samples), output_path)
                
            except Exception as e:
                print(f"[!] {lang_name} error: {e}")

        return None # We no longer return a collective dataset

    def harvest_magicoder(self, limit=110000):
        print(f"\n[🌟] Harvesting Magicoder-Evol (Logic Base)...")
        try:
            magic_ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train", streaming=True)
            all_samples = []
            pbar = tqdm(total=limit, desc="Collecting Logic", unit=" sample")
            for ex in magic_ds:
                if len(all_samples) >= limit: break
                all_samples.append({
                    "text": f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['response']}"
                })
                pbar.update(1)
            pbar.close()
            return Dataset.from_list(all_samples)
        except Exception as e:
            print(f"[!] Magicoder error: {e}")
            return None

    def save_dataset(self, dataset, path):
        if not dataset or len(dataset) == 0: return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dataset.to_json(path, orient="records", lines=True, force_ascii=False)
        print(f"[SUCCESS] Saved: {path} ({len(dataset)} items)")
