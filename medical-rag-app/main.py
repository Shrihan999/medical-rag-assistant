#!/usr/bin/env python3
"""
Main entry point for the Medical RAG application.
Run with different arguments to perform different functions:
- python main.py --preprocess: Download and preprocess the dataset
- python main.py --index: Generate embeddings and create vector index
- python main.py --download-model: Download the medical LLM model
- python main.py --streamlit: Launch the Streamlit web interface
- python main.py --all: Perform all preprocessing steps
"""

import os
import argparse
import logging
import json
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
    from src.data_processor import preprocess_dataset, download_dataset
    from src.indexer import generate_embeddings, create_vector_index
    from src.llm_interface import download_model
    from src.utils import create_directories
    MODULE_IMPORTS = True
except ImportError:
    logger.warning("Could not import project modules. Using built-in functions.")
    MODULE_IMPORTS = False

# Create necessary directories if module imports failed
if not MODULE_IMPORTS:
    def create_directories():
        """Create the necessary directories for the project."""
        directories = [
            "data/raw",
            "data/processed/embeddings",
            "models/Med-Qwen2-7B-GGUF",
            "streamlit_app",
            "src"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")

    def download_dataset():
        """Download the medical QA dataset from Hugging Face."""
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
        """Process the downloaded dataset for RAG."""
        try:
            from datasets import load_from_disk
            import pandas as pd
            
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

    def download_model():
        """Download the medical LLM model from Hugging Face."""
        try:
            from huggingface_hub import hf_hub_download
            
            logger.info("Downloading Med-Qwen2-7B-GGUF model from Hugging Face...")
            model_id = "mradermacher/Med-Qwen2-7B-GGUF"
            
            # First try to download just the model info to verify the filename
            logger.info("Checking model files...")
            try:
                from huggingface_hub import list_repo_files
                files = list_repo_files(model_id)
                gguf_files = [f for f in files if f.endswith(".gguf")]
                if gguf_files:
                    model_filename = gguf_files[0]
                    logger.info(f"Found model file: {model_filename}")
                else:
                    logger.warning("No .gguf files found. Using fallback name.")
                    model_filename = "Med-Qwen2-7b.Q4_K_M.gguf"
            except Exception as e:
                logger.warning(f"Could not list repo files: {e}")
                model_filename = "Med-Qwen2-7b.Q4_K_M.gguf"
            
            # Download the model
            output_path = "models/Med-Qwen2-7B-GGUF"
            model_path = hf_hub_download(
                repo_id=model_id,
                filename=model_filename,
                local_dir=output_path,
            )
            
            logger.info(f"Model downloaded and saved to {model_path}")
            
            # Save model info
            model_info = {
                "model_id": model_id,
                "model_filename": model_filename,
                "model_path": model_path
            }
            with open("models/model_info.json", "w") as f:
                json.dump(model_info, f, indent=2)
            
            return model_path
        
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return None

    def generate_embeddings():
        """Generate embeddings for the processed QA documents."""
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
            
            # Load processed data
            logger.info("Loading processed QA documents...")
            try:
                with open("data/processed/qa_documents.json", "r") as f:
                    documents = json.load(f)
            except FileNotFoundError:
                logger.warning("Processed documents not found. Running preprocessing first...")
                documents = preprocess_dataset()
                if documents is None:
                    return None
            
            # Load embedding model
            logger.info("Loading embedding model...")
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(documents)} documents...")
            texts = [doc["content"] for doc in documents]
            
            # Process in batches to avoid memory issues
            batch_size = 1000
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_end = min(i + batch_size, len(texts))
                logger.info(f"Processing batch {i} to {batch_end}...")
                batch_texts = texts[i:batch_end]
                batch_embeddings = embedding_model.encode(batch_texts, show_progress_bar=True)
                all_embeddings.append(batch_embeddings)
            
            # Combine batches
            embeddings = np.vstack(all_embeddings)
            
            # Save embeddings
            logger.info("Saving embeddings...")
            embeddings_path = "data/processed/embeddings/document_embeddings.npy"
            np.save(embeddings_path, embeddings)
            
            # Save document IDs for retrieval
            doc_ids = [doc["metadata"]["id"] for doc in documents]
            with open("data/processed/embeddings/document_ids.json", "w") as f:
                json.dump(doc_ids, f)
            
            # Save embedding metadata
            embedding_metadata = {
                "model_name": "all-MiniLM-L6-v2",
                "dimension": embeddings.shape[1],
                "num_documents": len(documents)
            }
            with open("data/processed/embeddings/metadata.json", "w") as f:
                json.dump(embedding_metadata, f, indent=2)
            
            logger.info(f"Embeddings generated and saved to {embeddings_path}")
            return embeddings
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return None

    def create_vector_index():
        """Create a FAISS vector index for the embeddings."""
        try:
            import numpy as np
            import faiss
            
            # Load embeddings
            logger.info("Loading embeddings...")
            try:
                embeddings = np.load("data/processed/embeddings/document_embeddings.npy")
            except FileNotFoundError:
                logger.warning("Embeddings not found. Generating embeddings first...")
                embeddings = generate_embeddings()
                if embeddings is None:
                    return None
            
            # Create FAISS index
            logger.info("Creating FAISS index...")
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
            index.add(embeddings.astype('float32'))
            
            # Save index
            index_path = "data/processed/embeddings/faiss_index.bin"
            faiss.write_index(index, index_path)
            
            logger.info(f"FAISS index created and saved to {index_path}")
            return index
        
        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            return None

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
    parser = argparse.ArgumentParser(description="Medical RAG Application")
    parser.add_argument("--preprocess", action="store_true", help="Download and preprocess the dataset")
    parser.add_argument("--index", action="store_true", help="Generate embeddings and create vector index")
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
        logger.info("=== Starting dataset preprocessing ===")
        preprocess_dataset()
    
    if args.all or args.index:
        logger.info("=== Starting embedding generation and indexing ===")
        generate_embeddings()
        create_vector_index()
    
    if args.all or args.download_model:
        logger.info("=== Starting model download ===")
        download_model()
    
    if args.streamlit:
        logger.info("=== Starting Streamlit application ===")
        launch_streamlit()

if __name__ == "__main__":
    main()