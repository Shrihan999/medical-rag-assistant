"""
Utility functions for the Medical RAG application.
"""

import logging
import os
from pathlib import Path
import subprocess
import sys
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def create_directories():
    """Create the necessary directories for the project."""
    directories = [
        "data/raw",
        "data/processed/embeddings",
        "models/Med-Qwen2-7B-GGUF",
        "streamlit_app",
        "src",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def log_query(query, answer, documents=None, success=True):
    """
    Log a user query and the system's response.
    
    Args:
        query (str): The user's query
        answer (str): The system's answer
        documents (list, optional): Retrieved documents
        success (bool): Whether the query was successfully processed
    """
    try:
        Path("logs").mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "query": query,
            "answer": answer,
            "success": success
        }
        
        if documents:
            log_entry["num_documents"] = len(documents)
            log_entry["top_document_score"] = documents[0]["score"] if documents else None
        
        # Append to log file
        with open("logs/query_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        logger.info(f"Query logged: {query[:50]}...")
    
    except Exception as e:
        logger.error(f"Error logging query: {e}")

def check_requirements():
    """
    Check if all required packages are installed.
    
    Returns:
        bool: True if all requirements are met, False otherwise
    """
    try:
        required_packages = [
            "numpy",
            "pandas",
            "sentence-transformers",
            "faiss-cpu",
            "streamlit",
            "llama-cpp-python",
            "huggingface-hub",
            "datasets"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.warning(f"Missing required packages: {', '.join(missing_packages)}")
            logger.info("Installing missing packages...")
            
            for package in missing_packages:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    logger.info(f"Installed {package}")
                except subprocess.CalledProcessError:
                    logger.error(f"Failed to install {package}")
                    return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error checking requirements: {e}")
        return False
def check_file_status():
    """Check the existence of required files and return their status."""
    files = {
        "processed_documents": {"path": "data/processed/qa_documents.json", "exists": False},
        "faiss_index": {"path": "data/processed/embeddings/faiss_index.bin", "exists": False},
        "model_file": {"path": "models/Med-Qwen2-7B-GGUF", "exists": False},  # Check folder instead
    }

    for key, file_info in files.items():
        file_info["exists"] = os.path.exists(file_info["path"])
        print(f"Checking {file_info['path']}: {file_info['exists']}")  # Debug output

    return files



if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Test utility functions
    create_directories()
    check_requirements()