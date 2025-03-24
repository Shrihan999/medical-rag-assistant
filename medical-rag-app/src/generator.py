"""
Generator module for the Medical RAG application.
Handles generation of answers based on retrieved context.
"""

import logging
import os
from pathlib import Path

from src.retriever import Retriever
from src.llm_interface import load_model, generate_answer

logger = logging.getLogger(__name__)

class Generator:
    """
    Generator class for generating answers to medical questions.
    """
    
    def __init__(self):
        """Initialize the generator."""
        self.retriever = Retriever()
        self.model = None
        
        # Lazy loading - will load the model when needed
    
    def _ensure_model_loaded(self):
        """Ensure the model is loaded."""
        if self.model is None:
            logger.info("Loading LLM model...")
            self.model = load_model()
            if self.model is None:
                logger.error("Failed to load LLM model")
                return False
            logger.info("LLM model loaded successfully")
        return True
    
    def answer_question(self, question, top_k=5, max_tokens=512, temperature=0.1):
        """
        Answer a medical question using RAG.
        
        Args:
            question (str): The medical question
            top_k (int): Number of documents to retrieve
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            
        Returns:
            dict: Dictionary containing the answer, retrieved documents, and metadata
        """
        try:
            # Step 1: Retrieve relevant documents
            logger.info(f"Retrieving documents for question: {question[:50]}...")
            retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
            
            if not retrieved_docs:
                logger.warning("No documents retrieved")
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "documents": [],
                    "success": False
                }
            
            # Step 2: Format context
            context = self.retriever.format_retrieved_context(retrieved_docs)
            
            # Step 3: Generate answer
            if self._ensure_model_loaded():
                logger.info("Generating answer...")
                answer = generate_answer(
                    self.model, 
                    question, 
                    context, 
                    max_tokens=max_tokens, 
                    temperature=temperature
                )
            else:
                return {
                    "answer": "Error: LLM model could not be loaded.",
                    "documents": retrieved_docs,
                    "success": False
                }
            
            return {
                "answer": answer,
                "documents": retrieved_docs,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "documents": [],
                "success": False
            }

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Test generator
    generator = Generator()
    result = generator.answer_question("What are the symptoms of diabetes?")
    
    print("\nQuestion: What are the symptoms of diabetes?")
    print("\nAnswer:")
    print(result["answer"])
    
    print("\nRetrieved Documents:")
    for i, doc in enumerate(result["documents"]):
        print(f"Document {i+1} (Score: {doc['score']:.4f}):")
        print(f"Question: {doc['metadata']['question'][:100]}...")
        print(f"Answer: {doc['metadata']['answer'][:100]}...")
        print()