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
├── config/             # YAML configurations
├── data/               # Local datasets & vector DB (Git-ignored)
│   └── rag/
│       └── domains/    # Hierarchical Raw/Processed data per domain
├── src/                # Implementation
│   ├── core/           # Business Logic
│   │   ├── models/     # Dataclasses
│   │   ├── SecretAssistant.py # Hybrid RAG AI
│   │   ├── RAGEngine.py       # Vector & Keyword Search
│   ├── data/           # Data Engineering
│   │   ├── harvesters/ # Specialized Domain Harvesters (SOLID)
│   │   ├── HarvesterOrchestrator.py # Multi-domain pipeline
│   │   └── KnowledgeProcessor.py    # Deduplication & Cleaning
│   └── training/       # Unsloth Training
├── SecretAiApp.py      # Main CLI Entry point
└── requirements.txt    # Project dependencies
```

## ⚙️ Configuration-First Workflow

SecretAi is 100% configuration-driven. All parameters (Hybrid search weights, LoRA targets, HF datasets, deduplication rules) are managed via `config/config.yaml`.

## 🚀 How to Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Setup Environment**:
   Create a `.env` file with your `HF_TOKEN`.
3. **Run the Pipeline**:
   - **Harvest Data**: `python SecretAiApp.py --mode harvest`
   - **Process & Dedup**: `python SecretAiApp.py --mode process`
   - **Index to RAG**: `python SecretAiApp.py --mode index`
   - **Chat with AI**: `python SecretAiApp.py --mode chat`
   - **Train Model**: `python SecretAiApp.py --mode train`

---
*Developed by M. Fatih Çelik as part of the SecretAi Research Initiative.*
