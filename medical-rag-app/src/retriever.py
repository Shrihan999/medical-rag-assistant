import os
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

class MedicalRAGRetriever:
    def __init__(self, 
                 index_path: str = "data/processed/medical_rag_index.faiss",
                 processed_dir: str = "data/processed",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Medical RAG Retriever.
        
        Args:
            index_path (str): Path to the FAISS index
            processed_dir (str): Directory with processed text files
            embedding_model (str): Sentence transformer model name
        """
        self.index_path = index_path
        self.processed_dir = processed_dir
        
        # Load embedding model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        
        # Load FAISS index
        try:
            self.index = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index from {index_path}")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            self.index = None
        
        # Load text files mapping
        self.load_text_files()
    
    def load_text_files(self):
        """
        Load processed text files to create a mapping for retrieval.
        """
        self.text_files = {}
        self.sentence_mapping = []
        
        for filename in os.listdir(self.processed_dir):
            if filename.endswith("_processed.txt"):
                file_path = os.path.join(self.processed_dir, filename)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    sentences = f.readlines()
                
                doc_name = os.path.splitext(filename)[0]
                self.text_files[doc_name] = sentences
                self.sentence_mapping.extend([(doc_name, i) for i in range(len(sentences))])
    
    def retrieve_relevant_contexts(self, 
                                   query: str, 
                                   top_k: int = 5, 
                                   similarity_threshold: float = 0.5):
        """
        Retrieve most relevant contexts for a given query.
        
        Args:
            query (str): Input query
            top_k (int): Number of top contexts to retrieve
            similarity_threshold (float): Minimum similarity score
        
        Returns:
            List[Dict]: List of relevant contexts with metadata
        """
        if self.index is None:
            logger.error("FAISS index not loaded")
            return []
        
        # Embed the query
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
        
        # Perform similarity search
        try:
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Process retrieved results
            relevant_contexts = []
            for dist, idx in zip(distances[0], indices[0]):
                # Convert similarity (inner product) to actual similarity score
                similarity = dist
                
                if similarity >= similarity_threshold:
                    # Get document and sentence info
                    doc_name, sentence_idx = self.sentence_mapping[idx]
                    context = {
                        'document': doc_name,
                        'sentence': self.text_files[doc_name][sentence_idx].strip(),
                        'similarity': similarity
                    }
                    relevant_contexts.append(context)
            
            # Sort by similarity in descending order
            relevant_contexts.sort(key=lambda x: x['similarity'], reverse=True)
            
            return relevant_contexts
        
        except Exception as e:
            logger.error(f"Error retrieving contexts: {e}")
            return []