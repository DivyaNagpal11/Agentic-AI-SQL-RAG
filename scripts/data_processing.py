"""
Data processing utilities for handling PDF and CSV files.
"""
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import csv
import sqlite3
import json
import argparse
import warnings

from config import PDF_DIR, CSV_DIR, SQLITE_DB_PATH

# Import the PDF loading components
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from unstructured.staging.base import elements_to_json, elements_from_json
from unstructured.staging.base import convert_to_dict
from unstructured.partition.pdf import partition_pdf

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PdfLoader:
    def __init__(self) -> None:
        self.loader_dict = {
            "PyPDFLoader": self.load_PyPDF,
            "UnstructuredPDFFromLangChain": self.load_UnstructuredPDFFromLangChain,
            "Unstructured": self.load_unstructured,
        }

    def load_PyPDF(self, file_name: str) -> List:
        """
        Load data using PyPDFLoader.
        """
        logger.info(f"Processing PDF file with PyPDFLoader: {file_name}")
        try:
            loader = PyPDFLoader(file_name, extract_images=True)
            pages = loader.load()
            for page in pages:
                logger.debug(f"Page: {page.metadata['page']}")
                logger.debug(page.page_content)
            return pages
        except Exception as e:
            logger.error(f"Error processing {file_name} with PyPDFLoader: {str(e)}")
            return []

    def load_UnstructuredPDFFromLangChain(self, file_name: str) -> List:
        """
        Load data using UnstructuredPDFLoader in LangChain.
        """
        logger.info(f"Processing PDF file with UnstructuredPDFFromLangChain: {file_name}")
        try:
            loader = UnstructuredPDFLoader(file_name, mode="elements")
            elements = loader.load()
            return elements
        except Exception as e:
            logger.error(f"Error processing {file_name} with UnstructuredPDFFromLangChain: {str(e)}")
            return []

    def load_unstructured(self, file_name: str) -> List:
        """
        Load data using unstructured.
        Requirements:
        - brew install poppler
        - brew install tesseract
        """
        logger.info(f"Processing PDF file with Unstructured: {file_name}")
        try:
            elements = partition_pdf(
                filename=file_name,
                infer_table_structure=True,
                strategy='ocr_only',
            )
            return elements
        except Exception as e:
            logger.error(f"Error processing {file_name} with Unstructured: {str(e)}")
            # Try with fallback settings
            logger.info(f"Attempting with fallback settings...")
            try:
                elements = partition_pdf(
                    filename=file_name,
                    infer_table_structure=False,
                    strategy='fast',
                    extract_images_in_pdf=False,
                    use_ocr=False
                )
                return elements
            except Exception as e2:
                logger.error(f"Fallback processing failed for {file_name}: {str(e2)}")
                return []

    def load(self, file_name: str, loader_name="Unstructured") -> List:
        """
        Load data using specified loader.
        """
        if loader_name in self.loader_dict:
            self.loader_name = loader_name
            result = self.loader_dict[self.loader_name](file_name)
            return result
        else:
            logger.error(f"Loader {loader_name} not found. Available loaders: {list(self.loader_dict.keys())}")
            return []

    def save_elements_json(self, elements, filename, json_path):
        """
        Save elements to JSON file in specified path.
        """
        try:
            convert_to_dict(elements)
            output = os.path.join(json_path, os.path.splitext(os.path.basename(filename))[0] + ".json")
            elements_to_json(elements, filename=output)
            logger.info(f"Elements saved to {output}")
            return output
        except Exception as e:
            logger.error(f"Error saving elements to JSON: {str(e)}")
            return None

    def save_elements(self, elements, filename):
        """
        Save elements to JSON file.
        """
        try:
            convert_to_dict(elements)
            output = filename + ".json"
            elements_to_json(elements, filename=output)
            logger.info(f"Elements saved to {output}")
            return output
        except Exception as e:
            logger.error(f"Error saving elements to JSON: {str(e)}")
            return None

    def load_elements_json(self, filename):
        """
        Load elements from JSON file.
        """
        try:
            elements = elements_from_json(filename=filename)
            logger.info(f"Elements loaded from {filename}")
            return elements
        except Exception as e:
            logger.error(f"Error loading elements from JSON: {str(e)}")
            return []

def process_pdf_file(pdf_path: Path, loader_name="Unstructured") -> List[Dict[str, Any]]:
    """
    Process a single PDF file using the PdfLoader class.
    
    Args:
        pdf_path: Path to the PDF file
        loader_name: Name of the loader to use
        
    Returns:
        List of document elements with metadata
    """
    logger.info(f"Processing PDF file: {pdf_path}")
    try:
        pdf_loader = PdfLoader()
        elements = pdf_loader.load(str(pdf_path), loader_name=loader_name)
        
        # Add metadata to each element if not already present
        processed_elements = []
        for i, element in enumerate(elements):
            # Check if element is already a dictionary with metadata
            if hasattr(element, 'metadata'):
                # Element from langchain has different structure
                if hasattr(element, 'page_content'):
                    processed_elements.append({
                        'content': element.page_content,
                        'metadata': {
                            'source': pdf_path.name,
                            'page_number': element.metadata.get('page', None),
                            'element_id': f"{pdf_path.stem}-{i}",
                            'element_type': "text",
                            'filename': pdf_path.name
                        }
                    })
                else:
                    # Element from unstructured
                    processed_elements.append({
                        'content': str(element),
                        'metadata': {
                            'source': pdf_path.name,
                            'page_number': getattr(element.metadata, 'page_number', None) if hasattr(element, 'metadata') else None,
                            'element_id': f"{pdf_path.stem}-{i}",
                            'element_type': element.category if hasattr(element, 'category') else "unknown",
                            'filename': pdf_path.name
                        }
                    })
            else:
                # Element is already a dictionary or other object
                processed_elements.append({
                    'content': str(element),
                    'metadata': {
                        'source': pdf_path.name,
                        'element_id': f"{pdf_path.stem}-{i}",
                        'filename': pdf_path.name
                    }
                })
        
        logger.info(f"Successfully processed {pdf_path} - extracted {len(processed_elements)} elements")
        
        # Optionally save the raw elements
        json_output = pdf_loader.save_elements(elements, str(pdf_path.with_suffix('')))
        logger.info(f"Raw elements saved to {json_output}")
        
        return processed_elements
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        return []

def process_all_pdfs(pdf_directory: Optional[Path] = None, loader_name="Unstructured") -> List[Dict[str, Any]]:
    """
    Process all PDF files in the specified directory.
    
    Args:
        pdf_directory: Directory containing PDF files. Defaults to config.PDF_DIR.
        loader_name: Name of the PDF loader to use
        
    Returns:
        List of all document elements from all PDFs
    """
    if pdf_directory is None:
        pdf_directory = PDF_DIR
    
    logger.info(f"Processing all PDFs in {pdf_directory}")
    
    all_elements = []
    
    # Check if directory exists
    if not pdf_directory.exists():
        logger.error(f"PDF directory does not exist: {pdf_directory}")
        return all_elements
    
    # Process each PDF file
    for pdf_file in pdf_directory.glob("*.pdf"):
        elements = process_pdf_file(pdf_file, loader_name=loader_name)
        all_elements.extend(elements)
    
    logger.info(f"Completed processing all PDFs. Total elements: {len(all_elements)}")
    return all_elements

def process_csv_file(csv_path: Path) -> pd.DataFrame:
    """
    Process a single CSV file and return a pandas DataFrame.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame containing the CSV data
    """
    logger.info(f"Processing CSV file: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully processed {csv_path} - shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error processing {csv_path}: {str(e)}")
        return pd.DataFrame()

def infer_schema_from_csv(csv_path: Path) -> List[Dict[str, str]]:
    """
    Infer schema (column names and types) from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of dictionaries with column name and inferred SQL type
    """
    df = process_csv_file(csv_path)
    if df.empty:
        return []
    
    schema = []
    for column in df.columns:
        # Simple type inference logic - can be expanded
        if pd.api.types.is_numeric_dtype(df[column]):
            if pd.api.types.is_integer_dtype(df[column]):
                col_type = "INTEGER"
            else:
                col_type = "REAL"
        elif pd.api.types.is_datetime64_dtype(df[column]):
            col_type = "TIMESTAMP"
        else:
            col_type = "TEXT"
        
        schema.append({"name": column, "type": col_type})
    
    return schema

def get_table_name_from_file(file_path: Path) -> str:
    """Generate a SQL-friendly table name from a file path."""
    # Remove extension and replace non-alphanumeric chars with underscore
    import re
    table_name = re.sub(r'[^a-zA-Z0-9]', '_', file_path.stem)
    # Ensure it starts with letter
    if not table_name[0].isalpha():
        table_name = 'table_' + table_name
    return table_name.lower()

def process_all_csvs(csv_directory: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Process all CSV files in the specified directory.
    
    Args:
        csv_directory: Directory containing CSV files. Defaults to config.CSV_DIR.
        
    Returns:
        Dictionary mapping table names to DataFrames
    """
    if csv_directory is None:
        csv_directory = CSV_DIR
    
    logger.info(f"Processing all CSVs in {csv_directory}")
    
    tables = {}
    
    # Check if directory exists
    if not csv_directory.exists():
        logger.error(f"CSV directory does not exist: {csv_directory}")
        return tables
    
    # Process each CSV file
    for csv_file in csv_directory.glob("*.csv"):
        table_name = get_table_name_from_file(csv_file)
        df = process_csv_file(csv_file)
        if not df.empty:
            tables[table_name] = df
    
    logger.info(f"Completed processing all CSVs. Total tables: {len(tables)}")
    return tables

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDF and CSV files.")
    parser.add_argument("--pdf_dir", type=str, help="Directory containing PDF files to process.")
    parser.add_argument("--csv_dir", type=str, help="Directory containing CSV files to process.")
    parser.add_argument("--pdf_loader", type=str, default="Unstructured", 
                      choices=["PyPDFLoader", "UnstructuredPDFFromLangChain", "Unstructured"],
                      help="PDF loader to use (default: Unstructured)")
    parser.add_argument("output_file", type=str, help="Path to the output JSON file.")
    
    args = parser.parse_args()
    
    output_data = {}
    
    if args.pdf_dir:
        pdf_path = Path(args.pdf_dir)
        pdf_elements = process_all_pdfs(pdf_path, loader_name=args.pdf_loader)
        output_data["pdf_elements"] = pdf_elements
    
    if args.csv_dir:
        csv_path = Path(args.csv_dir)
        csv_tables = process_all_csvs(csv_path)
        # Convert DataFrames to a serializable format (e.g., list of dicts)
        output_data["csv_tables"] = {
            table_name: df.to_dict(orient="records")
            for table_name, df in csv_tables.items()
        }
    
    # Save all output data to a JSON file
    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Processed data and saved to {args.output_file}")