"""
Indexer module for the Medical RAG application.
Handles embedding generation and vector index creation.
"""

import logging
import json
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_embeddings():
    """
    Generate embeddings for the processed QA documents.
    
    Returns:
        numpy.ndarray: The generated embeddings or None if an error occurred
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load processed data
        logger.info("Loading processed QA documents...")
        try:
            with open("data/processed/qa_documents.json", "r") as f:
                documents = json.load(f)
        except FileNotFoundError:
            logger.warning("Processed documents not found. Please run preprocessing first.")
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
        Path("data/processed/embeddings").mkdir(parents=True, exist_ok=True)
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
    """
    Create a FAISS vector index for the embeddings.
    
    Returns:
        faiss.Index: The created index or None if an error occurred
    """
    try:
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

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Generate embeddings
    embeddings = generate_embeddings()
    
    # Create index
    if embeddings is not None:
        create_vector_index()