import os
import logging
import numpy as np
import torch
from typing import List
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger(__name__)

def load_embeddings_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load sentence embedding model.
    
    Args:
        model_name (str): Name of the embedding model
    
    Returns:
        SentenceTransformer: Loaded embedding model
    """
    try:
        # Check for GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name, device=device)
        logger.info(f"Loaded embedding model on {device}")
        return model
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        return None

def generate_embeddings(processed_dir: str = "data/processed", 
                        embeddings_dir: str = "data/processed/embeddings"):
    """
    Generate embeddings for processed documents.
    
    Args:
        processed_dir (str): Directory with processed text files
        embeddings_dir (str): Directory to save embeddings
    """
    # Create embeddings directory if it doesn't exist
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Load embedding model
    model = load_embeddings_model()
    if model is None:
        logger.error("Failed to load embedding model")
        return
    
    # Process each file in the processed directory
    for filename in os.listdir(processed_dir):
        if filename.endswith("_processed.txt"):
            file_path = os.path.join(processed_dir, filename)
            
            try:
                # Read sentences from the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    sentences = f.readlines()
                
                # Generate embeddings
                embeddings = model.encode(sentences, show_progress_bar=True)
                
                # Save embeddings
                embeddings_path = os.path.join(embeddings_dir, f"{os.path.splitext(filename)[0]}_embeddings.npy")
                np.save(embeddings_path, embeddings)
                
                logger.info(f"Generated embeddings for {filename}")
            
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")

def create_vector_index(embeddings_dir: str = "data/processed/embeddings", 
                        index_path: str = "data/processed/medical_rag_index.faiss"):
    """
    Create a FAISS vector index from embeddings.
    
    Args:
        embeddings_dir (str): Directory containing embedding files
        index_path (str): Path to save the FAISS index
    """
    # Collect all embeddings
    all_embeddings = []
    embedding_files = [f for f in os.listdir(embeddings_dir) if f.endswith("_embeddings.npy")]
    
    if not embedding_files:
        logger.error("No embedding files found")
        return
    
    # Load all embeddings
    for filename in embedding_files:
        file_path = os.path.join(embeddings_dir, filename)
        embeddings = np.load(file_path)
        all_embeddings.append(embeddings)
    
    # Concatenate embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Determine embedding dimension
    embedding_dim = all_embeddings.shape[1]
    
    # Create FAISS index
    try:
        # Create an index using cosine similarity
        index = faiss.IndexFlatIP(embedding_dim)  # Inner product is equivalent to cosine similarity
        
        # Convert to float32 if needed
        all_embeddings = all_embeddings.astype('float32')
        
        # Add embeddings to the index
        index.add(all_embeddings)
        
        # Save the index
        faiss.write_index(index, index_path)
        
        logger.info(f"Created vector index at {index_path}")
        logger.info(f"Total vectors in index: {index.ntotal}")
    
    except Exception as e:
        logger.error(f"Error creating vector index: {e}")