import logging
from typing import List, Dict
from src.llm_interface import load_medical_llm
from src.retriever import MedicalRAGRetriever

logger = logging.getLogger(__name__)

class MedicalRAGGenerator:
    def __init__(self, 
                 retriever: MedicalRAGRetriever = None, 
                 max_context_tokens: int = 1024,
                 max_response_tokens: int = 512):
        """
        Initialize the Medical RAG Generator.
        
        Args:
            retriever (MedicalRAGRetriever): RAG retriever instance
            max_context_tokens (int): Maximum tokens for context
            max_response_tokens (int): Maximum tokens for response
        """
        self.retriever = retriever or MedicalRAGRetriever()
        self.model = load_medical_llm()
        
        self.max_context_tokens = max_context_tokens
        self.max_response_tokens = max_response_tokens
    
    def generate_response(self, 
                          query: str, 
                          num_contexts: int = 3, 
                          similarity_threshold: float = 0.5) -> str:
        """
        Generate a response using retrieved medical contexts.
        
        Args:
            query (str): User's medical query
            num_contexts (int): Number of contexts to retrieve
            similarity_threshold (float): Minimum similarity threshold
        
        Returns:
            str: Generated medical response
        """
        if self.model is None:
            logger.error("Medical LLM not loaded")
            return "Sorry, the medical model is currently unavailable."
        
        # Retrieve relevant medical contexts
        contexts = self.retriever.retrieve_relevant_contexts(
            query, 
            top_k=num_contexts, 
            similarity_threshold=similarity_threshold
        )
        
        # Prepare the prompt with retrieved contexts
        context_text = "\n".join([
            f"[Context {i+1} from {ctx['document']}]: {ctx['sentence']}" 
            for i, ctx in enumerate(contexts)
        ])
        
        # Construct the full prompt
        prompt = f"""You are a medical AI assistant. 
        Given the following medical contexts and the user's query, provide a helpful, accurate, and safe response.

Relevant Medical Contexts:
{context_text}

User Query: {query}

Response:"""
        
        try:
            # Generate response using the medical LLM
            response = self.model(
                prompt, 
                max_new_tokens=self.max_response_tokens
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm unable to generate a response at the moment."