import os
import logging
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)

def save_json(data: Dict[str, Any], filepath: str):
    """
    Save data to a JSON file.
    
    Args:
        data (Dict): Data to be saved
        filepath (str): Path to save the JSON file
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved data to {filepath}")
    except Exception as e:
        logger.error(f"Error saving JSON file: {e}")

def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        filepath (str): Path to the JSON file
    
    Returns:
        Dict: Loaded data or empty dict if error
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {filepath}: {e}")
        return {}

def create_project_structure():
    """
    Create the basic project directory structure.
    """
    project_dirs = [
        "data/documents",
        "data/processed",
        "data/processed/embeddings",
        "models/Med-Qwen2-7B-GGUF",
        "streamlit_app"
    ]
    
    for directory in project_dirs:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")