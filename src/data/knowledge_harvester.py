import requests
from bs4 import BeautifulSoup
import json
import time
import re
import sys
import os
import yaml
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
from src.utils.config_manager import config_manager
from src.core.models import ScrapedPage

class KnowledgeHarvester:
    """
    KnowledgeHarvester handles the web scraping of technical documentation
    to enrich the RAG knowledge base.
    """
    def __init__(self):
        self.config = config_manager
        self.seeds_path = self.config.get('paths.harvest_seeds', 'config/harvest_seeds.yaml')
        self.seeds = self._load_seeds()
        
        # Load categories from config
        self.primary_cat = self.config.get('categories.primary', 'Backend')
        self.secondary_cat = self.config.get('categories.secondary', 'Android')
        
        self._setup_output_encoding()

    def _setup_output_encoding(self):
        if sys.platform == "win32":
            import codecs
            sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

    def _load_seeds(self):
        if not os.path.exists(self.seeds_path):
            print(f"[WARNING] Seeds config not found at {self.seeds_path}. Using empty seeds.")
            return {}
        with open(self.seeds_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("seeds", {})

    def _clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _get_links_shallow(self, base_url):
        try:
            response = requests.get(base_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = []
            for a in soup.find_all('a', href=True):
                full_url = urljoin(base_url, a['href'])
                if full_url.startswith(base_url):
                    links.append(full_url)
            return list(set(links))
        except Exception as e:
            print(f"[ERROR] Failed to get links from {base_url}: {e}")
            return []

    def _scrape_page(self, task) -> ScrapedPage:
        url, category = task
        try:
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            content = self._clean_text(soup.get_text())
            
            # Extract basic code snippets
            code_snippets = [code.get_text() for code in soup.find_all('code') if len(code.get_text()) > 20]
            
            return ScrapedPage(
                category=category,
                topic=soup.title.string if soup.title else "Technical Doc",
                content=content,
                code_snippets=code_snippets[:3],
                source=url
            )
        except Exception:
            return None

    def run(self):
        """
        Executes the harvesting process based on the seeds configuration.
        """
        all_tasks = []
        print("🔍 Collecting links from documentation seeds...")
        
        for category, seeds in self.seeds.items():
            for seed in seeds:
                links = self._get_links_shallow(seed)
                # Apply limits based on category for balanced enrichment
                # Using config-driven logic for limits too would be better, but category check is now safer
                limit = 60 if category == self.primary_cat else 30
                for link in links[:limit]:
                    all_tasks.append((link, category))
        
        print(f"🚀 Starting parallel mining for {len(all_tasks)} pages...")
        
        new_data = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(self._scrape_page, all_tasks))
            for res in results:
                if res:
                    new_data.append(res.to_dict())
                
        # Append to the master knowledge base file
        master_file = self.config.get('paths.raw_data', 'data/knowledge_base_massive.json')
        os.makedirs(os.path.dirname(master_file), exist_ok=True)
        
        existing_data = []
        if os.path.exists(master_file):
            with open(master_file, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        
        combined_data = existing_data + new_data
        
        with open(master_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Success: {len(new_data)} new pages harvested and saved to {master_file}")

if __name__ == "__main__":
    harvester = KnowledgeHarvester()
    harvester.run()
