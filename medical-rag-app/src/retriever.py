"""
Retriever module for the Medical RAG application.
Handles retrieval of relevant documents based on query.
"""

import logging
import json
import numpy as np
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class Retriever:
    """
    Retriever class for finding relevant documents for a given query.
    """
    
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        """
        Initialize the retriever.
        
        Args:
            embedding_model_name (str): Name of the embedding model to use
        """
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.index = None
        self.documents = None
        self.document_ids = None
        
        # Load components if available
        self._load_components()
    
    def _load_components(self):
        """Load the necessary components for retrieval."""
        try:
            # Check if index exists
            if not os.path.exists("data/processed/embeddings/faiss_index.bin"):
                logger.warning("FAISS index not found. Please create the index first.")
                return False
            
            # Check if embeddings metadata exists
            if not os.path.exists("data/processed/embeddings/metadata.json"):
                logger.warning("Embeddings metadata not found. Please generate embeddings first.")
                return False
            
            # Check if document IDs exist
            if not os.path.exists("data/processed/embeddings/document_ids.json"):
                logger.warning("Document IDs not found. Please generate embeddings first.")
                return False
            
            # Check if documents exist
            if not os.path.exists("data/processed/qa_documents.json"):
                logger.warning("Processed documents not found. Please preprocess the dataset first.")
                return False
            
            # Load document IDs
            with open("data/processed/embeddings/document_ids.json", "r") as f:
                self.document_ids = json.load(f)
            
            # Load documents
            with open("data/processed/qa_documents.json", "r") as f:
                self.documents = json.load(f)
            
            # Load index
            import faiss
            self.index = faiss.read_index("data/processed/embeddings/faiss_index.bin")
            
            # Load embedding model
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            logger.info("Retriever components loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading retriever components: {e}")
            return False
    
    def retrieve(self, query, top_k=5):
        """
        Retrieve the most relevant documents for a query.
        
        Args:
            query (str): The query text
            top_k (int): Number of documents to retrieve
            
        Returns:
            list: List of retrieved documents with their metadata
        """
        try:
            if self.embedding_model is None or self.index is None or self.documents is None:
                logger.warning("Retriever components not loaded. Trying to load...")
                success = self._load_components()
                if not success:
                    logger.error("Failed to load retriever components")
                    return []
            
            # Encode the query
            query_embedding = self.embedding_model.encode([query])[0].reshape(1, -1).astype('float32')
            
            # Search the index
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Get the documents
            retrieved_docs = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.document_ids):
                    doc_id = self.document_ids[idx]
                    # Find the document with this ID
                    doc = next((d for d in self.documents if d["metadata"]["id"] == doc_id), None)
                    if doc:
                        # Add the distance/similarity score
                        doc["score"] = float(distances[0][i])
                        retrieved_docs.append(doc)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}...")
            return retrieved_docs
        
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def format_retrieved_context(self, docs):
        """
        Format retrieved documents into a context string.
        
        Args:
            docs (list): List of retrieved documents
            
        Returns:
            str: Formatted context string
        """
        context = ""
        for i, doc in enumerate(docs):
            context += f"[Document {i+1}]\n"
            context += f"Question: {doc['metadata']['question']}\n"
            context += f"Answer: {doc['metadata']['answer']}\n\n"
        
        return context.strip()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Test retriever
    retriever = Retriever()
    test_query = "What are the symptoms of diabetes?"
    docs = retriever.retrieve(test_query, top_k=3)
    
    if docs:
        print(f"Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs):
            print(f"Document {i+1} (Score: {doc['score']:.4f}):")
            print(f"Question: {doc['metadata']['question'][:100]}...")
            print(f"Answer: {doc['metadata']['answer'][:100]}...")
            print()
    else:
        print("No documents retrieved.")