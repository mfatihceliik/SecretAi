from datasets import load_dataset, Dataset
from typing import Optional
from tqdm.auto import tqdm
from .DatasetProcessor import DatasetProcessor
from .models import TrainingSample
from src.utils.ConfigManager import config_manager
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
    """
    SecretAiDatasetLoader handles downloading and preparing datasets,
    driven by configuration settings.
    """
    def __init__(self, tokenizer=None):
        self.config = config_manager
        self.tokenizer = tokenizer
        self.processor = DatasetProcessor()
        self.hf_token = os.getenv("HF_TOKEN")
        
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

    def process_stack_example(self, example, lang_name) -> Optional[TrainingSample]:
        code = self._download_s3_content(example.get("blob_id"), example.get("src_encoding"))
        if not code or len(str(code)) < 500: return None
        
        example["content"] = code
        if not self.processor.is_good_code(example, lang_name): return None
            
        return self.processor.format_sample(code, lang_name, tokenizer=self.tokenizer)

    def harvest_stack_v2(self, samples_per_lang=20000):
        languages = self.processor.lang_config_manager.languages
        print(f"--- SecretAi RESILIENT HARVESTER (Stack-v2) ---")
        
        max_workers = 10 
        datasets_dir = self.config.get('paths.training_datasets_dir', 'data/training/datasets')

        for lang_cfg in languages:
            lang_name = lang_cfg["name"]
            output_path = os.path.join(datasets_dir, f"stack_{lang_name}.jsonl")
            
            if os.path.exists(output_path):
                print(f"[INFO] {lang_name} already exists, skipping: {output_path}")
                continue

            print(f"[🚀] Harvesting {lang_name}...")
            
            try:
                subset_stream = load_dataset("bigcode/the-stack-v2", name=lang_name, split="train", streaming=True, token=self.hf_token)
                lang_samples = []
                scanned_count = 0
                pbar = tqdm(total=samples_per_lang, desc=f"Collecting {lang_name}")

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
                            processed_sample = future.result()
                            if processed_sample:
                                lang_samples.append(processed_sample.to_dict())
                                pbar.update(1)
                            
                            if len(lang_samples) < samples_per_lang and scanned_count < 1000000:
                                try:
                                    ex = next(stream_iter)
                                    futures.add(executor.submit(self.process_stack_example, ex, lang_name))
                                    scanned_count += 1
                                except StopIteration: pass

                pbar.close()
                
                if lang_samples:
                    self.save_dataset(Dataset.from_list(lang_samples), output_path)
                
            except Exception as e:
                print(f"[!] {lang_name} error: {e}")

    def harvest_magicoder(self, limit=110000):
        print(f"[🌟] Harvesting Magicoder-Evol...")
        try:
            magic_ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train", streaming=True)
            all_samples = []
            pbar = tqdm(total=limit, desc="Logic Extraction")
            for ex in magic_ds:
                if len(all_samples) >= limit: break
                
                instruction = ex['instruction']
                response = ex['response']
                
                sample = TrainingSample(text=f"### Instruction:\n{instruction}\n\n### Response:\n{response}")
                
                all_samples.append(sample.to_dict())
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
        print(f"[SUCCESS] Saved {len(dataset)} items to {path}")
