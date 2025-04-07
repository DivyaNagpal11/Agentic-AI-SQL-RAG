"""
Create text chunks for the RAG system using the by_title strategy.
"""
import logging
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.chunking.title import chunk_by_title
import json
import argparse  # Import argparse

from config import CHUNK_SIZE, CHUNK_OVERLAP

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_chunks_by_title(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chunk documents using the by_title strategy from Unstructured.
    
    Args:
        documents: List of document elements with content and metadata
        
    Returns:
        List of chunked documents
    """
    logger.info(f"Creating chunks using by_title strategy for {len(documents)} documents")
    
    try:
        # Extract raw elements
        elements = []
        element_metadata = {}
        
        for doc in documents:
            element_id = doc["metadata"]["element_id"]
            elements.append(doc["content"])
            element_metadata[element_id] = doc["metadata"]
        
        # Use Unstructured's chunk_by_title
        chunked_elements = chunk_by_title(
            elements=elements,
            max_characters=CHUNK_SIZE,
            new_after_n_chars=CHUNK_SIZE,
            combine_text_under_n_chars=CHUNK_OVERLAP
        )
        
        # Reconstruct documents with metadata
        chunked_docs = []
        for i, chunk in enumerate(chunked_elements):
            # Find original metadata or use generic
            # This is a simplification - in practice, you'd track which original element each chunk came from
            original_id = f"element-{i}"  # Placeholder
            metadata = element_metadata.get(original_id, {}).copy()
            
            # Update metadata for the chunk
            metadata.update({
                "chunk_id": f"chunk-{i}",
                "chunk_size": len(chunk),
                "is_chunk": True
            })
            
            chunked_docs.append({
                "content": chunk,
                "metadata": metadata
            })
        
        logger.info(f"Created {len(chunked_docs)} chunks")
        return chunked_docs
        
    except Exception as e:
        logger.error(f"Error creating chunks: {str(e)}")
        return documents  # Return original documents if chunking fails

def create_chunks_by_text_splitter(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Alternative chunking method using LangChain's RecursiveCharacterTextSplitter.
    
    Args:
        documents: List of document elements with content and metadata
        
    Returns:
        List of chunked documents
    """
    logger.info(f"Creating chunks using text splitter for {len(documents)} documents")
    
    try:
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False
        )
        
        chunked_docs = []
        for i, doc in enumerate(documents):
            # Extract text and metadata
            text = doc["content"]
            metadata = doc["metadata"].copy()
            
            # Split text into chunks
            texts = text_splitter.split_text(text)
            
            # Create new documents with updated metadata
            for j, chunk_text in enumerate(texts):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_id": f"{metadata.get('element_id', f'doc-{i}')}-chunk-{j}",
                    "chunk_size": len(chunk_text),
                    "chunk_index": j,
                    "total_chunks": len(texts),
                    "is_chunk": True
                })
                
                chunked_docs.append({
                    "content": chunk_text,
                    "metadata": chunk_metadata
                })
        
        logger.info(f"Created {len(chunked_docs)} chunks")
        return chunked_docs
        
    except Exception as e:
        logger.error(f"Error creating chunks with text splitter: {str(e)}")
        return documents  # Return original documents if chunking fails

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk documents for RAG.")
    parser.add_argument("input_file", type=str, help="Path to the input JSON file containing documents.")
    parser.add_argument("output_file", type=str, help="Path to the output JSON file to save chunks.")
    parser.add_argument("--method", type=str, default="title", choices=["title", "splitter"], help="Chunking method: title or splitter.")

    args = parser.parse_args()

    # Load documents from JSON
    with open(args.input_file, 'r') as f:
        documents = json.load(f)

    # Chunk documents
    if args.method == "title":
        chunked_docs = create_chunks_by_title(documents)
    else:
        chunked_docs = create_chunks_by_text_splitter(documents)

    # Save chunks to JSON
    with open(args.output_file, 'w') as f:
        json.dump(chunked_docs, f, indent=4)

    print(f"Created {len(chunked_docs)} chunks and saved to {args.output_file}")