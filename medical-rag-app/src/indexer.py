import logging
import pickle
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils import EMBEDDING_MODEL_NAME, INDEX_PATH, DOCSTORE_PATH, EMBEDDINGS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Indexer:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """Initializes the Indexer with an embedding model."""
        try:
            self.embedding_model = SentenceTransformer(model_name)
            logging.info(f"Loaded embedding model: {model_name}")
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            logging.info(f"Embedding dimension: {self.dimension}")
        except Exception as e:
            logging.error(f"Failed to load embedding model {model_name}: {e}")
            raise

    def create_index(self, chunks: List[Dict[str, str]]):
        """
        Creates a FAISS index from the provided text chunks.
        Saves the index and the corresponding chunk data.
        """
        if not chunks:
            logging.warning("No chunks provided to create index.")
            return

        logging.info(f"Starting index creation for {len(chunks)} chunks...")

        # Extract page content for embedding
        texts = [chunk["page_content"] for chunk in chunks]

        # Generate embeddings
        logging.info("Generating embeddings...")
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            embeddings = np.array(embeddings).astype('float32') # FAISS requires float32
            logging.info(f"Generated {len(embeddings)} embeddings.")
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            return

        # Create FAISS index
        # Using IndexFlatL2 - simple L2 distance search
        # For larger datasets, consider more advanced index types like IndexIVFFlat
        index = faiss.IndexFlatL2(self.dimension)
        index.add(embeddings)
        logging.info(f"FAISS index created. Total vectors in index: {index.ntotal}")

        # Save the index and the chunks (docstore)
        try:
            EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            faiss.write_index(index, str(INDEX_PATH))
            logging.info(f"FAISS index saved to {INDEX_PATH}")

            # Save the corresponding chunks (mapping index ID to chunk)
            with open(DOCSTORE_PATH, "wb") as f:
                pickle.dump(chunks, f)
            logging.info(f"Document chunks (docstore) saved to {DOCSTORE_PATH}")

        except Exception as e:
            logging.error(f"Error saving index or docstore: {e}")

def build_index(processed_chunks: List[Dict[str, str]]):
    """Helper function to build and save the index."""
    if not processed_chunks:
        logging.error("Cannot build index: No processed chunks provided.")
        return

    indexer = Indexer()
    indexer.create_index(processed_chunks)

if __name__ == '__main__':
    # Example Usage: Assumes you have processed chunks (e.g., from data_processor)
    # In a real script (like main.py), you'd pass the output of process_documents here.

    # Dummy data for testing standalone
    dummy_chunks = [
        {"page_content": "The quick brown fox jumps over the lazy dog.", "metadata": {"source": "dummy1.txt"}},
        {"page_content": "Medical diagnosis requires careful examination.", "metadata": {"source": "dummy2.txt"}},
        {"page_content": "Artificial intelligence is transforming healthcare.", "metadata": {"source": "dummy3.txt"}},
    ]
    print("Building index with dummy data...")
    build_index(dummy_chunks)
    print(f"Index built. Check {INDEX_PATH} and {DOCSTORE_PATH}")

    # Test loading (similar to what retriever would do)
    if INDEX_PATH.exists() and DOCSTORE_PATH.exists():
        print("\nTesting index loading...")
        try:
            index = faiss.read_index(str(INDEX_PATH))
            with open(DOCSTORE_PATH, "rb") as f:
                docstore = pickle.load(f)
            print(f"Successfully loaded index with {index.ntotal} vectors.")
            print(f"Successfully loaded docstore with {len(docstore)} documents.")
        except Exception as e:
            print(f"Error loading index/docstore: {e}")
    else:
        print("\nIndex files not found, skipping loading test.")