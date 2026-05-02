import requests
from bs4 import BeautifulSoup
import json
import time
import re
import sys
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
import os
from src.utils.config_manager import config_manager

if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# --- BACKEND AND ANDROID ENRICHMENT TARGETS ---
HARVEST_SEEDS = {
    "Backend": [
        # Python
        "https://docs.python.org/3/tutorial/index.html",
        "https://fastapi.tiangolo.com/tutorial/",
        "https://docs.djangoproject.com/en/stable/intro/",
        
        # JavaScript / Node.js
        "https://nodejs.org/en/learn/getting-started/introduction-to-nodejs",
        "https://expressjs.com/en/guide/routing.html",
        "https://docs.nestjs.com/",
        
        # Java & Libraries
        "https://dev.java/learn/",
        "https://hibernate.org/orm/documentation/6.4/",
        "https://junit.org/junit5/docs/current/user-guide/",
        
        # Spring Depth
        "https://docs.spring.io/spring-framework/reference/core.html"
    ],
    "Android": [
        "https://developer.android.com/training/data-storage/room",
        "https://developer.android.com/topic/libraries/architecture/datastore",
        "https://ktor.io/docs/client-getting-started.html", # Ktor Client for Android
        "https://developer.android.com/jetpack/compose/state"
    ]
}

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_links_shallow(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = set()
        domain = url.split('/')[2]
        for a in soup.find_all('a', href=True):
            full_url = urljoin(url, a['href']).split('#')[0]
            if domain in full_url and "/reference/" not in full_url:
                links.add(full_url)
        return list(links)
    except: return []

def scrape_page(url_and_cat):
    url, category = url_and_cat
    try:
        print(f"HARVESTING: {url}")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=12)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for extra in soup(['nav', 'footer', 'header', 'script', 'style', 'aside']):
            extra.decompose()
            
        title = soup.title.string.strip() if soup.title else url
        content_area = soup.find('article') or soup.find('main') or soup.body
        if not content_area: return None

        code_snippets = [c.get_text().strip() for c in content_area.find_all(['pre', 'code']) if len(c.get_text()) > 40]
        
        return {
            "category": category,
            "topic": title,
            "content": clean_text(content_area.get_text(separator=' ')),
            "code_snippets": code_snippets[:15],
            "source": url
        }
    except: return None

def run_backend_powerup():
    all_tasks = []
    print("🔍 Collecting new Backend and Android links...")
    for category, seeds in HARVEST_SEEDS.items():
        for seed in seeds:
            links = get_links_shallow(seed)
            # Fetch more pages for Backend (for enrichment)
            limit = 60 if category == "Backend" else 30
            for link in links[:limit]:
                all_tasks.append((link, category))
    
    print(f"🚀 Starting parallel mining for {len(all_tasks)} new pages...")
    
    new_data = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(scrape_page, all_tasks))
        for res in results:
            if res: new_data.append(res)
            
    # Read existing file and append (Append logic)
    master_file = config_manager.get('paths.raw_data', 'data/knowledge_base_massive.json')
    if os.path.exists(master_file):
        with open(master_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            existing_data.extend(new_data)
            final_data = existing_data
    else:
        final_data = new_data
            
    with open(master_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n🏆 POWER-UP SUCCESS: {len(new_data)} new docs added to {master_file}.")

if __name__ == "__main__":
    run_backend_powerup()
