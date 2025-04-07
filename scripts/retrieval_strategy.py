"""
Implement retrieval strategies for the RAG system.
"""
import logging
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
import sqlite3

from scripts.config import (
    CHROMA_DB_PATH, 
    SQLITE_DB_PATH, 
    TOP_K, 
    SIMILARITY_THRESHOLD,
    USE_OLLAMA
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_embedding_function():
    """Initialize and return the embedding function based on configuration."""
    if USE_OLLAMA:
        try:
            from langchain_community.embeddings import OllamaEmbeddings
            return OllamaEmbeddings(model="llama3")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama embeddings: {e}")
            logger.info("Falling back to HuggingFace embeddings")
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    else:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def retrieve_from_vector_db(
    query: str,
    collection_name: str = "documents",
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    db_path: Optional[str] = None
) -> List[Document]:
    """
    Retrieve relevant documents from ChromaDB based on the query.
    
    Args:
        query: User query string
        collection_name: Name of the ChromaDB collection
        top_k: Number of documents to retrieve. Defaults to config.TOP_K.
        threshold: Similarity threshold. Defaults to config.SIMILARITY_THRESHOLD.
        db_path: Path to ChromaDB. Defaults to config.CHROMA_DB_PATH.
        
    Returns:
        List of retrieved Document objects
    """
    if top_k is None:
        top_k = TOP_K
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD
    if db_path is None:
        db_path = str(CHROMA_DB_PATH)
    
    logger.info(f"Retrieving documents for query: '{query}'")
    
    try:
        # Initialize embedding function
        embedding_function = initialize_embedding_function()
        
        # Load Chroma vector store
        db = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_function,
            collection_name=collection_name
        )
        
        # Retrieve documents with MMR (Maximum Marginal Relevance) for diversity
        documents = db.max_marginal_relevance_search(
            query=query,
            k=top_k,
            fetch_k=top_k * 2,  # Fetch more docs initially for better diversity
            lambda_mult=0.5  # Balance between relevance and diversity
        )
        
        # Filter by similarity threshold if needed
        # Note: MMR doesn't return similarity scores directly, so we'd need to do a separate similarity search
        # This is a simplified approach
        if threshold > 0:
            results_with_scores = db.similarity_search_with_score(query, k=top_k)
            threshold_docs = [doc for doc, score in results_with_scores if score >= threshold]
            
            # Combine MMR results with threshold filtering
            doc_ids = set([doc.metadata.get("element_id") for doc in documents])
            for doc in threshold_docs:
                if doc.metadata.get("element_id") not in doc_ids:
                    documents.append(doc)
            
            # Limit back to top_k
            documents = documents[:top_k]
        
        logger.info(f"Retrieved {len(documents)} documents")
        return documents
        
    except Exception as e:
        logger.error(f"Error retrieving from vector DB: {str(e)}")
        return []

def execute_sql_query(
    query: str,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute a SQL query against the SQLite database.
    
    Args:
        query: SQL query string
        db_path: Path to SQLite database. Defaults to config.SQLITE_DB_PATH.
        
    Returns:
        Dictionary with query results and metadata
    """
    if db_path is None:
        db_path = str(SQLITE_DB_PATH)
    
    logger.info(f"Executing SQL query: {query}")
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        
        # Execute query
        df = pd.read_sql_query(query, conn)
        
        # Get metadata about the query
        cursor = conn.cursor()
        tables_used = []
        
        # Extract table names from query (simple approach)
        # A more robust approach would use a SQL parser
        query_upper = query.upper()
        if "FROM" in query_upper:
            from_parts = query_upper.split("FROM")[1].split()
            if len(from_parts) > 0:
                tables_used.append(from_parts[0].strip().rstrip(";,"))
        
        # Close connection
        conn.close()
        
        logger.info(f"Query returned {len(df)} rows")
        
        return {
            "results": df.to_dict(orient="records"),
            "columns": list(df.columns),
            "row_count": len(df),
            "tables_used": tables_used,
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Error executing SQL query: {str(e)}")
        return {
            "error": str(e),
            "query": query
        }

def get_database_schema(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the schema of the SQLite database.
    
    Args:
        db_path: Path to SQLite database. Defaults to config.SQLITE_DB_PATH.
        
    Returns:
        Dictionary with database schema information
    """
    if db_path is None:
        db_path = str(SQLITE_DB_PATH)
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Get schema for each table
        schema = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = [
                {
                    "name": row[1],
                    "type": row[2],
                    "notnull": bool(row[3]),
                    "pk": bool(row[4])
                }
                for row in cursor.fetchall()
            ]
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table};")
            row_count = cursor.fetchone()[0]
            
            # Store table info
            schema[table] = {
                "columns": columns,
                "row_count": row_count
            }
        
        # Close connection
        conn.close()
        
        return {
            "tables": tables,
            "schema": schema
        }
        
    except Exception as e:
        logger.error(f"Error getting database schema: {str(e)}")
        return {"error": str(e)}

def hybrid_retrieval(
    query: str,
    collection_name: str = "documents",
    top_k: Optional[int] = None,
    chroma_db_path: Optional[str] = None,
    sqlite_db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform hybrid retrieval from both vector and SQL databases.
    
    Args:
        query: User query string
        collection_name: Name of the ChromaDB collection
        top_k: Number of documents to retrieve. Defaults to config.TOP_K.
        chroma_db_path: Path to ChromaDB. Defaults to config.CHROMA_DB_PATH.
        sqlite_db_path: Path to SQLite database. Defaults to config.SQLITE_DB_PATH.
        
    Returns:
        Dictionary with retrieval results from both sources
    """
    logger.info(f"Performing hybrid retrieval for query: '{query}'")
    
    # Get vector search results
    vector_docs = retrieve_from_vector_db(
        query=query,
        collection_name=collection_name,
        top_k=top_k,
        db_path=chroma_db_path
    )
    
    # Get database schema for context
    db_schema = get_database_schema(db_path=sqlite_db_path)
    
    return {
        "vector_results": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in vector_docs
        ],
        "database_schema": db_schema
    }

if __name__ == "__main__":
    # Test retrieval functions
    test_query = "What are the key features of the product?"
    
    # Test vector retrieval
    vector_docs = retrieve_from_vector_db(test_query)
    print(f"Retrieved {len(vector_docs)} documents from vector DB")
    
    # Test SQL schema retrieval
    schema = get_database_schema()
    print(f"Database schema: {len(schema.get('tables', [])) if isinstance(schema, dict) else 'Error'} tables")
    
    # Test hybrid retrieval
    hybrid_results = hybrid_retrieval(test_query)
    print(f"Hybrid retrieval results: {len(hybrid_results.get('vector_results', []))} vector docs")