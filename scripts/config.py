"""
Configuration settings for the Q&A chatbot project.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdf_directory"
CSV_DIR = DATA_DIR / "csv_directory"
DB_DIR = PROJECT_ROOT / "db"

# Create directories if they don't exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Database settings
SQLITE_DB_PATH = DB_DIR / "structured_data.db"
CHROMA_DB_PATH = DB_DIR / "chroma_vector_db"

# LLM settings
# Use local Ollama model or HuggingFace model
USE_OLLAMA = True  # Set to False to use HuggingFace
OLLAMA_MODEL = "llama3"  # or any other model you have pulled
HUGGINGFACE_MODEL = "google/flan-t5-large"

# For optional OpenAI integration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
TOP_K = 5
SIMILARITY_THRESHOLD = 0.7

# Evaluation settings
EVALUATION_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]