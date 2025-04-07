# Q&A Chatbot with RAG and SQL Agents

A powerful chatbot that combines Retrieval-Augmented Generation (RAG) and SQL agents to answer questions from both unstructured documents and structured databases. It features smart query routing, hybrid retrieval, and seamless integration with local and cloud-based LLMs.

---

## 🚀 Overview

This project implements a multi-agent natural language Q&A system that can:

- 🧾 Process **unstructured data** (PDFs) using **RAG**
- 🧮 Query **structured data** (CSV → SQLite) using **SQL**
- 🧠 Intelligently **route queries** to the correct agent based on intent
- 💬 Provide **context-aware, accurate answers** using advanced language models

The system uses **Unstructured.io** for document parsing and **ChromaDB** for vector search. A primary agent delegates user queries to specialized agents.

---

## 🏗️ Architecture

```text
            ┌─────────────┐
            │   User      │
            │   Query     │
            └──────┬──────┘
                   │
        ┌──────────▼──────────┐
        │     Primary Agent   │
        │  (Query Classifier) │
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼───────┐    ┌────────▼────────┐
│   RAG Agent   │    │    SQL Agent    │
│ (Unstructured │    │   (Structured   │
│  Data: PDFs)  │    │      Data)      │
└───────┬───────┘    └────────┬────────┘
        │ Retrieve Docs         │ Execute SQL
┌───────▼───────┐    ┌────────▼────────┐
│  ChromaDB     │    │     SQLite      │
│(Vector Store) │    │   Database      │
└───────────────┘    └─────────────────┘
                   │
        ┌──────────▲──────────┐
        │     Primary Agent   │
        │  (Generate Response)│
        └──────────┬──────────┘
                   │
        ┌───────▼───────┐
        │     User      │
        │    Answer     │
        └───────────────┘
```

---

## ✨ Features

- **Multi-agent Architecture** – Specialized RAG and SQL agents improve precision.
- **Hybrid Retrieval** – Combines vector search and SQL for rich answers.
- **Smart Chunking** – Uses Unstructured.io's `by_title` for semantically coherent document chunks.
- **Evaluation via RAGAS** – Automated response quality scoring.
- **Gradio UI** – Intuitive, web-based chat interface.
- **LLM Flexibility** – Plug-and-play support for Hugging Face and Ollama (local models, preferred way).

---

## 📁 Project Structure

```text
project/
├── requirements.txt
├── config.py
├── data/
│   ├── pdf_directory/
│   └── csv_directory/
├── scripts/
│   ├── data_processing.py
│   ├── db_storage.py
│   ├── chunk_creation.py
│   ├── retrieval_strategy.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── primary_agent.py
│   │   ├── rag_agent.py
│   │   └── sql_agent.py
│   ├── evaluation.py
│   ├── report_generation.py
│   └── app.py
└── README.md
```

---

## ⚙️ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your/repo-name.git
   cd repo-name
   ```

2. **(Optional)** Create a virtual environment  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional)** Install Ollama for local LLMs  
   ```bash
   brew install ollama
   ollama serve
   ollama pull llama3
   ```

---

## 🔧 Configuration

Edit `config.py` to customize:

- 📂 Data directories
- 🗄️ Database paths
- 🤖 LLM settings (Ollama vs Hugging Face)
- 📐 Chunking parameters
- 🔍 Retrieval strategy
- 📊 Evaluation metrics

---

## 📥 Adding & Processing Data

### ➕ Unstructured Data (PDFs)
Place PDFs in: `data/pdf_directory/`

### ➕ Structured Data (CSVs)
Place CSVs in: `data/csv_directory/`  
(*Ensure first row has column headers*)

### 🔄 Process All Data
```bash
python scripts/data_processing.py --pdf_dir data/pdf_directory --csv_dir data/csv_directory output.json
```

### 💾 Store Processed Data
```bash
python scripts/db_storage.py output.json --sqlite_table my_data --chroma_collection my_collection
```

---

## 🖥️ Running the App

Start the Gradio web interface:

```bash
python scripts/app.py
```

Visit the link in the terminal (usually `http://127.0.0.1:7860`).


---

## 🧩 Key Components

### 🧱 Chunking Strategy

Implemented in `chunk_creation.py`, using Unstructured.io’s `by_title` approach:

- Recognizes sections by headers
- Creates semantically meaningful chunks
- Preserves document hierarchy

### 🔍 Retrieval Strategy

Defined in `retrieval_strategy.py`:

- Semantic search via ChromaDB
- Maximum Marginal Relevance (MMR)
- Customizable thresholds

### 🧠 Agent System

- `PrimaryAgent` – Routes queries
- `RAGAgent` – Handles unstructured docs
- `SQLAgent` – Handles tabular data

---

## 📊 Evaluation with RAGAS

Evaluate chatbot responses on metrics like:

- **Faithfulness** (factual accuracy)
- **Answer Relevance**
- **Context Precision/Recall**


## 🤝 Contributing

Pull requests are welcome!  
Suggestions for improvements, bug fixes, or new features are appreciated.

> Consider following a branch naming convention like `feature/`, `bugfix/`, or `docs/`.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
