import logging
import pickle
from typing import List, Dict, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils import EMBEDDING_MODEL_NAME, INDEX_PATH, DOCSTORE_PATH, TOP_K

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Retriever:
    def __init__(self,
                 index_path: str = str(INDEX_PATH),
                 docstore_path: str = str(DOCSTORE_PATH),
                 model_name: str = EMBEDDING_MODEL_NAME):
        """Initializes the Retriever, loading the index, docstore, and embedding model."""
        self.index_path = index_path
        self.docstore_path = docstore_path
        self.model_name = model_name
        self.embedding_model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.docstore: Optional[List[Dict[str, str]]] = None
        self._load()

    def _load(self):
        """Loads the FAISS index, docstore, and embedding model."""
        try:
            # Load embedding model
            self.embedding_model = SentenceTransformer(self.model_name)
            logging.info(f"Retriever: Loaded embedding model '{self.model_name}'")

            # Load FAISS index
            self.index = faiss.read_index(self.index_path)
            logging.info(f"Retriever: Loaded FAISS index from {self.index_path} ({self.index.ntotal} vectors)")

            # Load docstore (chunks)
            with open(self.docstore_path, "rb") as f:
                self.docstore = pickle.load(f)
            logging.info(f"Retriever: Loaded docstore from {self.docstore_path} ({len(self.docstore)} documents)")

            if self.index.ntotal != len(self.docstore):
                logging.warning(f"Index size ({self.index.ntotal}) and docstore size ({len(self.docstore)}) mismatch!")

        except FileNotFoundError as e:
            logging.error(f"Retriever: Error loading files: {e}. Index or docstore not found. Please run indexing first.")
            self.embedding_model = None
            self.index = None
            self.docstore = None
            # Consider raising an exception if loading is critical for initialization
            # raise RuntimeError("Failed to initialize Retriever due to missing index/docstore files.") from e
        except Exception as e:
            logging.error(f"Retriever: Error during initialization: {e}")
            self.embedding_model = None
            self.index = None
            self.docstore = None
            # raise RuntimeError("Failed to initialize Retriever.") from e

    def is_ready(self) -> bool:
        """Checks if the retriever is ready for querying."""
        return self.embedding_model is not None and self.index is not None and self.docstore is not None

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict[str, str]]:
        """
        Retrieves the top_k most relevant document chunks for a given query.
        Returns a list of chunk dictionaries.
        """
        if not self.is_ready():
            logging.error("Retriever is not ready. Index/docstore might be missing.")
            return []

        logging.info(f"Retrieving documents for query: '{query[:100]}...'") # Log snippet of query

        try:
            # 1. Embed the query
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True).astype('float32')

            # 2. Search the index
            # D = distances, I = indices
            distances, indices = self.index.search(query_embedding, top_k)

            # 3. Fetch the corresponding documents from docstore
            retrieved_docs = []
            if indices.size > 0:
                for i, idx in enumerate(indices[0]):
                    if 0 <= idx < len(self.docstore):
                        doc = self.docstore[idx]
                        doc['metadata']['score'] = float(distances[0][i]) # Add similarity score
                        retrieved_docs.append(doc)
                        # logging.debug(f"Retrieved doc {idx} with score {distances[0][i]}: {doc['page_content'][:100]}...")
                    else:
                         logging.warning(f"Invalid index {idx} found during retrieval.")

            logging.info(f"Retrieved {len(retrieved_docs)} documents.")
            return retrieved_docs

        except Exception as e:
            logging.error(f"Error during retrieval: {e}")
            return []

    def get_context_string(self, query: str, top_k: int = TOP_K) -> str:
        """Retrieves documents and formats them into a single context string."""
        retrieved_docs = self.retrieve(query, top_k)
        context = "\n\n".join([
            f"Source: {doc['metadata'].get('source', 'Unknown')}\nContent: {doc['page_content']}"
            for doc in retrieved_docs
        ])
        return context


if __name__ == '__main__':
    # Example Usage: Assumes index and docstore exist from running indexer.py
    print("Initializing Retriever...")
    try:
        retriever = Retriever()

        if retriever.is_ready():
            test_query = "hypertension treatment"
            print(f"\nTesting retrieval for query: '{test_query}'")
            results = retriever.retrieve(test_query)

            if results:
                print(f"\nRetrieved {len(results)} documents:")
                for i, doc in enumerate(results):
                    print(f"--- Result {i+1} ---")
                    print(f"Source: {doc['metadata'].get('source', 'N/A')}")
                    print(f"Score: {doc['metadata'].get('score', 'N/A'):.4f}")
                    print(f"Content: {doc['page_content'][:200]}...") # Show preview
            else:
                print("No documents retrieved. The index might be empty or the query unrelated.")

            print("\nTesting get_context_string:")
            context_string = retriever.get_context_string(test_query)
            print(f"Formatted Context String (first 500 chars):\n{context_string[:500]}...")

        else:
            print("\nRetriever initialization failed. Cannot run tests. Please ensure index/docstore files exist.")
            print(f"Expected index: {INDEX_PATH}")
            print(f"Expected docstore: {DOCSTORE_PATH}")

    except Exception as e:
        print(f"\nAn error occurred during standalone testing: {e}")
        print("Ensure index and docstore files exist and paths are correct in src/utils.py.")