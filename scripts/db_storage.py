"""
Store processed data in databases (SQLite for structured data, ChromaDB for vector storage)
"""
import sqlite3
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import json  # Import the json library
import argparse  # Import argparse

from scripts.config import (
    SQLITE_DB_PATH,
    CHROMA_DB_PATH,
    USE_OLLAMA,
    OLLAMA_MODEL,
    HUGGINGFACE_MODEL
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def store_dataframe_in_sqlite(df: pd.DataFrame, table_name: str, db_path: Optional[Path] = None) -> bool:
    """
    Store a pandas DataFrame in SQLite database.

    Args:
        df: DataFrame to store
        table_name: Name of the table to create/replace
        db_path: Path to SQLite database. Defaults to config.SQLITE_DB_PATH.

    Returns:
        True if successful, False otherwise
    """
    if db_path is None:
        db_path = SQLITE_DB_PATH

    logger.info(f"Storing DataFrame in SQLite table: {table_name}")

    try:
        # Create SQLite connection
        conn = sqlite3.connect(db_path)

        # Write the DataFrame to the database
        df.to_sql(table_name, conn, index=False, if_exists='replace')

        # Create a basic index on the first column
        first_col = df.columns[0]
        cursor = conn.cursor()
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{first_col} ON {table_name} ({first_col})")

        conn.commit()
        conn.close()

        logger.info(f"Successfully stored {len(df)} rows in table {table_name}")
        return True

    except Exception as e:
        logger.error(f"Error storing DataFrame in SQLite: {str(e)}")
        return False

def get_tables_in_sqlite(db_path: Optional[Path] = None) -> List[str]:
    """
    Get a list of all tables in the SQLite database.

    Args:
        db_path: Path to SQLite database. Defaults to config.SQLITE_DB_PATH.

    Returns:
        List of table names
    """
    if db_path is None:
        db_path = SQLITE_DB_PATH

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query for all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        conn.close()
        return tables

    except Exception as e:
        logger.error(f"Error getting tables from SQLite: {str(e)}")
        return []

def get_table_info(table_name: str, db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get information about a specific table in the SQLite database.

    Args:
        table_name: Name of the table
        db_path: Path to SQLite database. Defaults to config.SQLITE_DB_PATH.

    Returns:
        Dictionary with table schema and sample data
    """
    if db_path is None:
        db_path = SQLITE_DB_PATH

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get table schema
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [{"name": row[1], "type": row[2]} for row in cursor.fetchall()]

        # Get sample data (first 5 rows)
        df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 5", conn)
        sample_data = df.to_dict(orient='records')

        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]

        conn.close()

        return {
            "table_name": table_name,
            "columns": columns,
            "sample_data": sample_data,
            "row_count": row_count
        }

    except Exception as e:
        logger.error(f"Error getting info for table {table_name}: {str(e)}")
        return {"error": str(e)}

def initialize_embedding_function():
    """Initialize and return the embedding function based on configuration."""
    if USE_OLLAMA:
        try:
            return OllamaEmbeddings(model=OLLAMA_MODEL)
        except Exception as e:
            logger.error(f"Failed to initialize Ollama embeddings: {e}")
            logger.info("Falling back to HuggingFace embeddings")
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    else:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def store_documents_in_chroma(
    documents: List[Dict[str, Any]],
    collection_name: str = "documents",
    db_path: Optional[Path] = None
) -> bool:
    """
    Store document elements in ChromaDB.

    Args:
        documents: List of documents with 'content' and 'metadata' keys
        collection_name: Name of the collection in ChromaDB
        db_path: Path to ChromaDB. Defaults to config.CHROMA_DB_PATH.

    Returns:
        True if successful, False otherwise
    """
    if db_path is None:
        db_path = CHROMA_DB_PATH

    logger.info(f"Storing {len(documents)} documents in ChromaDB collection: {collection_name}")

    try:
        # Extract texts and metadata
        texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        ids = [doc["metadata"]["element_id"] for doc in documents]

        # Initialize embedding function
        embedding_function = initialize_embedding_function()

        # Create Chroma vector store
        db = Chroma.from_texts(
            texts=texts,
            embedding=embedding_function,
            metadatas=metadatas,
            ids=ids,
            persist_directory=str(db_path),
            collection_name=collection_name
        )

        # Persist to disk
        db.persist()

        logger.info(f"Successfully stored {len(documents)} documents in ChromaDB")
        return True

    except Exception as e:
        logger.error(f"Error storing documents in ChromaDB: {str(e)}")
        return False

def get_chroma_collection_info(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get information about ChromaDB collections.

    Args:
        db_path: Path to ChromaDB. Defaults to config.CHROMA_DB_PATH.

    Returns:
        Dictionary with collection information
    """
    if db_path is None:
        db_path = CHROMA_DB_PATH

    try:
        # Initialize client
        client = chromadb.PersistentClient(path=str(db_path))

        # Get all collections
        collections = client.list_collections()
        collection_names = [collection.name for collection in collections]

        # Get count for each collection
        collection_info = {}
        for name in collection_names:
            collection = client.get_collection(name)
            count = collection.count()
            collection_info[name] = {"document_count": count}

        return {
            "collections": collection_names,
            "collection_details": collection_info
        }

    except Exception as e:
        logger.error(f"Error getting ChromaDB collection info: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Store data in databases.")
    parser.add_argument("input_file", type=str, help="Path to the input JSON file.")
    parser.add_argument("--sqlite_table", type=str, help="Name of the SQLite table.")
    parser.add_argument("--chroma_collection", type=str, default="documents", help="Name of the ChromaDB collection.")

    args = parser.parse_args()

    # Load data from JSON
    try:
        with open(args.input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: Input JSON file '{args.input_file}' not found.")
        exit(1)  # Exit with an error code
    except json.JSONDecodeError:
        logger.error(f"Error: Invalid JSON format in '{args.input_file}'.")
        exit(1)

    # Handle different JSON structures for SQLite
    if args.sqlite_table:
        try:
            if isinstance(data, list):  # Array of objects
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and "table_data" in data:  # Object with table_data
                df = pd.DataFrame(data["table_data"])
            elif isinstance(data, dict) and "data" in data and "columns" in data:  # Object with data and columns
                df = pd.DataFrame(data["data"], columns=data["columns"])
            else:
                raise ValueError("Unsupported JSON structure for SQLite")
            store_dataframe_in_sqlite(df, args.sqlite_table)
        except ValueError as e:
            logger.error(f"Error processing JSON for SQLite: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error storing data in SQLite: {str(e)}")

    #  Handle ChromaDB storage (assumes input JSON is suitable for Chroma)
    if args.chroma_collection:
        try:
            if isinstance(data, list):
                store_documents_in_chroma(data, args.chroma_collection)
            else:
                logger.warning("ChromaDB input is not a list, it might cause issues.")
                store_documents_in_chroma([data], args.chroma_collection)

        except Exception as e:
            logger.error(f"Error storing data in ChromaDB: {str(e)}")