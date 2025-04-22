#!/usr/bin/env python3
"""
Main entry point for the Medical RAG application using PDF documents.
Run with different arguments to perform different functions:
- python main.py --preprocess: Process PDF documents in the data/documents folder
- python main.py --index: Generate embeddings and create vector index for PDFs
- python main.py --download-model: Download the medical LLM model
- python main.py --streamlit: Launch the Streamlit web interface
- python main.py --all: Perform all preprocessing steps
"""

import os
import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Import project modules
try:
    from src.data_processor import process_pdf_documents
    from src.indexer import generate_embeddings, create_vector_index
    from src.llm_interface import download_model, load_medical_llm
except ImportError as e:
    logger.error(f"Import error: {e}")
    process_pdf_documents = None
    generate_embeddings = None
    create_vector_index = None
    download_model = None
    load_medical_llm = None

def create_directories():
    """Create the necessary directories for the project."""
    directories = [
        "data/documents",
        "data/processed",
        "data/processed/embeddings",
        "models/Med-Qwen2-7B-GGUF",
        "streamlit_app"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def launch_streamlit():
    """Launch the Streamlit web interface."""
    try:
        # Check if streamlit_app/app.py exists
        if not os.path.exists("streamlit_app/app.py"):
            logger.error("streamlit_app/app.py not found. Cannot launch application.")
            return
            
        logger.info("Launching Streamlit application...")
        streamlit_cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app/app.py"]
        subprocess.run(streamlit_cmd)
    except Exception as e:
        logger.error(f"Error launching Streamlit: {e}")

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Medical RAG Application with PDF Documents")
    parser.add_argument("--preprocess", action="store_true", help="Process PDF documents in data/documents")
    parser.add_argument("--index", action="store_true", help="Generate embeddings and create vector index for PDFs")
    parser.add_argument("--download-model", action="store_true", help="Download the medical LLM model")
    parser.add_argument("--streamlit", action="store_true", help="Launch the Streamlit web interface")
    parser.add_argument("--all", action="store_true", help="Perform all preprocessing steps")
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # If no arguments provided, show help and exit
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Process arguments
    if args.all or args.preprocess:
        logger.info("=== Starting PDF document processing ===")
        if process_pdf_documents:
            process_pdf_documents()
        else:
            logger.error("PDF processing function not available")
    
    if args.all or args.index:
        logger.info("=== Starting PDF embedding generation and indexing ===")
        if generate_embeddings and create_vector_index:
            generate_embeddings()
            create_vector_index()
        else:
            logger.error("Embedding or indexing functions not available")
    
    if args.all or args.download_model:
        logger.info("=== Starting model download ===")
        if download_model:
            model_url = "https://huggingface.co/mradermacher/Med-Qwen2-7B-GGUF/resolve/main/Med-Qwen2-7B.Q4_K_S.gguf"
            download_model(model_url)
        else:
            logger.error("Model download function not available")
    
    if args.streamlit:
        logger.info("=== Starting Streamlit application ===")
        launch_streamlit()

if __name__ == "__main__":
    main()