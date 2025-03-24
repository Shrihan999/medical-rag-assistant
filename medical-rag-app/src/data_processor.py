"""
Data processing module for the Medical RAG application.
Handles dataset downloading and preprocessing.
"""

import logging
import json
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

def download_dataset():
    """
    Download the medical QA dataset from Hugging Face.
    
    Returns:
        dataset: The downloaded dataset or None if an error occurred
    """
    try:
        from datasets import load_dataset
        
        logger.info("Downloading medical QA dataset from Hugging Face...")
        dataset = load_dataset("Malikeh1375/medical-question-answering-datasets", "all-processed")
        
        # Save the dataset to disk
        dataset.save_to_disk("data/raw/medical_qa_dataset")
        logger.info("Dataset downloaded and saved to data/raw/medical_qa_dataset")
        
        # Save a sample for inspection
        sample_data = dataset["train"].select(range(min(5, len(dataset["train"])))).to_dict()
        with open("data/raw/sample_data.json", "w") as f:
            json.dump(sample_data, f, indent=2)
        logger.info("Sample data saved to data/raw/sample_data.json")
        
        return dataset
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return None

def preprocess_dataset():
    """
    Process the downloaded dataset for RAG.
    Converts the dataset into a format suitable for RAG.
    
    Returns:
        list: Processed documents or None if an error occurred
    """
    try:
        from datasets import load_from_disk
        
        logger.info("Loading dataset from disk...")
        try:
            dataset = load_from_disk("data/raw/medical_qa_dataset")
        except FileNotFoundError:
            logger.warning("Dataset not found on disk. Downloading first...")
            dataset = download_dataset()
            if dataset is None:
                return None
        
        logger.info("Processing dataset...")
        
        # Process the train split (the only one in this dataset)
        processed_data = []
        
        df = dataset["train"].to_pandas()
        logger.info(f"Processing {len(df)} records...")
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                instruction = str(row["instruction"]).strip()
                question = str(row["input"]).strip()
                answer = str(row["output"]).strip()
                
                # Skip empty entries
                if not question or not answer:
                    continue
                
                # Create document
                document = {
                    "content": f"Question: {question}\nAnswer: {answer}",
                    "metadata": {
                        "instruction": instruction,
                        "question": question,
                        "answer": answer,
                        "id": f"train_{idx}"
                    }
                }
                processed_data.append(document)
                
                # Log progress periodically
                if idx % 10000 == 0:
                    logger.info(f"Processed {idx} records...")
                    
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
        
        # Save processed data
        processed_file = "data/processed/qa_documents.json"
        with open(processed_file, "w") as f:
            json.dump(processed_data, f, indent=2)
        
        logger.info(f"Processed {len(processed_data)} QA pairs and saved to {processed_file}")
        return processed_data
    
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return None

def clean_text(text):
    """
    Clean and normalize text.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    return text

def explore_dataset(dataset_path=None):
    """
    Explore and analyze the dataset.
    
    Args:
        dataset_path (str, optional): Path to the dataset. If None, will try to load from default location.
        
    Returns:
        dict: Dataset statistics
    """
    try:
        from datasets import load_dataset, load_from_disk
        
        if dataset_path:
            try:
                dataset = load_from_disk(dataset_path)
            except:
                logger.info(f"Could not load from disk, trying to load from Hugging Face: {dataset_path}")
                dataset = load_dataset(dataset_path, "all-processed")
        else:
            try:
                dataset = load_from_disk("data/raw/medical_qa_dataset")
            except FileNotFoundError:
                logger.info("Dataset not found locally, downloading from Hugging Face...")
                dataset = load_dataset("Malikeh1375/medical-question-answering-datasets", "all-processed")
        
        stats = {}
        
        # Get dataset structure
        for split_name, split in dataset.items():
            stats[split_name] = {
                "num_examples": len(split),
                "features": list(split.features.keys()),
                "sample": split[0] if len(split) > 0 else None
            }
            
            # Column statistics
            for col in split.features:
                if col in split:
                    non_null = sum(1 for item in split[col] if item)
                    stats[split_name][f"{col}_stats"] = {
                        "non_null": non_null,
                        "null": len(split) - non_null
                    }
                    
                    # For string columns, get length statistics
                    if isinstance(split[col][0], str):
                        lengths = [len(str(item)) for item in split[col] if item]
                        if lengths:
                            stats[split_name][f"{col}_stats"].update({
                                "min_length": min(lengths),
                                "max_length": max(lengths),
                                "avg_length": sum(lengths) / len(lengths)
                            })
        
        # Save statistics
        with open("data/raw/dataset_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
            
        logger.info("Dataset exploration complete. Statistics saved to data/raw/dataset_stats.json")
        return stats
        
    except Exception as e:
        logger.error(f"Error exploring dataset: {e}")
        return None

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Create directories
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Explore the dataset
    explore_dataset()
    
    # Preprocess the dataset
    preprocess_dataset()