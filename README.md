# SecretAi: High-Performance Coding Assistant Pipeline

SecretAi is a specialized machine learning project focused on creating high-performance coding assistants through a hybrid data harvesting and fine-tuning pipeline. It leverages state-of-the-art techniques like **Unsloth** for optimized training and a custom filtering mechanism for high-quality dataset synthesis.

![SecretAi Architecture](assets/architecture.png)

## 🚀 Overview

The primary goal of SecretAi is to transform base LLMs (like Llama 3) into expert coding assistants. This is achieved by:
1.  **Hybrid Data Harvesting**: Combining structured logic (Magicoder) with massive raw source code (The Stack v2).
2.  **Intelligent Filtering**: Using configuration-driven thresholds and modern library detection to ensure only top-tier code is used for training.
3.  **RAG Integration**: Powering the assistant with a real-time vector database (ChromaDB) containing the latest documentation for Backend and Android development.
4.  **Optimized Fine-Tuning**: Using Unsloth for significantly faster and more memory-efficient training.

## 📁 Project Structure

```text
├── config/             # YAML configurations (config.yaml, harvest_seeds.yaml)
├── data/               # Local datasets & vector DB (Git-ignored)
├── src/                # Implementation
│   ├── core/           # Business Logic
│   │   ├── models/           # Data Models (ScrapedPage, KnowledgeChunk, etc.)
│   │   ├── secret_assistant.py # AI Interface
│   │   ├── rag_engine.py       # RAG Search & Indexing
│   │   ├── model_factory.py    # LLM Factory
│   │   ├── secret_ai_dataset_loader.py # Data Loading
│   │   └── dataset_processor.py # Filtering & Formatting
│   ├── training/       # Training Logic
│   │   └── secret_ai_trainer.py # Unsloth Fine-Tuning
│   ├── data/           # Data Engineering
│   │   ├── knowledge_harvester.py # Doc Scraping
│   │   ├── knowledge_processor.py # RAG Data Refinement
│   │   └── dataset_generator.py   # Dataset Orchestration
│   └── utils/          # Utilities (Config Management)
├── main.py             # CLI Entry point
└── requirements.txt    # Project dependencies
```

## ⚙️ Configuration-First Workflow

SecretAi is 100% configuration-driven. All parameters (LoRA targets, prompt templates, noise patterns, thresholds) are managed via `config/config.yaml`. No more hardcoded logic in the source code!

## 🚀 How to Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Setup Environment**:
   Create a `.env` file with your `HF_TOKEN`.
3. **Run the Application**:
   - **Chat Mode**: `python main.py --mode chat`
   - **Train Mode**: `python main.py --mode train`
   - **Index Mode**: `python main.py --mode index`
   - **Harvest Mode**: `python main.py --mode harvest`
   - **Generate Mode**: `python main.py --mode generate`

---
*Developed by M. Fatih Çelik as part of the SecretAi Research Initiative.*
