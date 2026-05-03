# 🚀 SecretAi Çalıştırma Rehberi

Bu rehber, SecretAi projesindeki karmaşık veri hattını (pipeline) hatasız bir şekilde çalıştırmanız için hazırlanmıştır.

## 📋 1. Ön Hazırlık
Öncelikle gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```
Proje kök dizininde `.env` dosyası oluşturun ve Hugging Face token'ınızı ekleyin:
```text
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

---

## 🛠️ 2. Veri Hattı (Pipeline) Akışı
Asistanın bilgi sahibi olması için aşağıdaki adımları **sırasıyla** takip etmelisiniz:

### ADIM 1: Veri Toplama (Harvesting)
Hugging Face üzerinden belirlenen domainlere (Kotlin, Java, Python, JS) ait verileri çeker ve `raw` klasörlerine kaydeder.
```bash
python SecretAiApp.py --mode harvest
```

### ADIM 2: Veri İşleme & Tekilleştirme (Processing)
Ham verileri temizler, **ContentHasher** ile kopyaları siler ve tüm domainleri `final_kb.json` dosyasında birleştirir.
```bash
python SecretAiApp.py --mode process
```

### ADIM 3: Vektör İndeksleme (Indexing)
İşlenmiş verileri **ChromaDB** vektör veritabanına yükleyerek asistanın "hafızasına" ekler.
```bash
python SecretAiApp.py --mode index
```

---

## 🤖 3. Kullanım Modları

### AI İle Sohbet (Chat)
RAG sistemi (Melez Arama) destekli asistanı başlatır:
```bash
python SecretAiApp.py --mode chat
```

### Model Eğitimi (Fine-Tuning)
Unsloth kullanarak yerel eğitim sürecini başlatır (GPU Gereklidir):
```bash
python SecretAiApp.py --mode train
```

---

## 📂 4. Önemli Dosya Yolları
- **Ham Veriler:** `data/rag/domains/{domain}/raw/`
- **İşlenmiş Veriler:** `data/rag/domains/{domain}/processed/`
- **Final Bilgi Bankası:** `data/rag/final/final_kb.json`
- **Veritabanı:** `data/rag/chroma_db/`

---
*İyi kodlamalar!*
