import logging
from src.retriever import Retriever
from src.llm_interface import generate_response, load_llm, download_model # Import necessary functions
from src.utils import TOP_K

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGGenerator:
    def __init__(self):
        """Initializes the RAG Generator, which orchestrates retrieval and generation."""
        logging.info("Initializing RAG Generator...")
        try:
            # Ensure model is available (download if needed, load on demand via llm_interface)
            if not download_model():
                 logging.warning("Model download check failed. LLM generation might not work if model is missing.")
            # Initialize retriever (loads index, docstore, embedding model)
            self.retriever = Retriever()
            if not self.retriever.is_ready():
                # Log error but allow initialization, generation will fail gracefully later
                logging.error("Retriever initialization failed. Context retrieval will not work.")
        except Exception as e:
            logging.error(f"Error during RAGGenerator initialization: {e}")
            self.retriever = None # Ensure retriever is None if it failed
            raise # Re-raise critical init errors

    def answer_query(self, query: str) -> str:
        """
        Answers a query using the RAG pipeline:
        1. Retrieves relevant context.
        2. Generates an answer using the LLM with the context.
        """
        logging.info(f"Processing query: '{query}'")

        # 1. Retrieve context
        context = ""
        if self.retriever and self.retriever.is_ready():
            context = self.retriever.get_context_string(query, top_k=TOP_K)
            if not context:
                logging.warning("No context retrieved for the query. Proceeding with query only (or LLM's internal knowledge).")
            else:
                 logging.info(f"Retrieved context (length: {len(context)} chars)")
        else:
            logging.warning("Retriever not available or not ready. Cannot retrieve context.")
            # Optionally, you could decide to *not* call the LLM if context is essential
            # return "Sorry, I cannot answer without access to the document index."

        # 2. Generate response using LLM
        try:
            # generate_response handles loading the LLM if needed
            final_answer = generate_response(query, context)
        except Exception as e:
            logging.error(f"Error during LLM response generation step: {e}")
            # Check if it's a loading error specifically
            if isinstance(e, FileNotFoundError) or "load_llm" in str(e):
                 final_answer = "Error: The language model file could not be found or loaded. Please ensure it is downloaded and configured correctly."
            else:
                final_answer = "Sorry, an error occurred while generating the answer."

        return final_answer

if __name__ == '__main__':
    # Example Usage: Assumes index, docstore, and model exist
    print("Initializing RAG Generator for standalone test...")
    try:
        rag_generator = RAGGenerator()

        test_query = "What are the main symptoms of diabetes type 2?"
        print(f"\nTesting RAG pipeline with query: '{test_query}'")

        answer = rag_generator.answer_query(test_query)

        print("\n--- Final Answer ---")
        print(answer)
        print("--------------------")

    except Exception as e:
        print(f"\nAn error occurred during RAG Generator standalone test: {e}")
        print("Ensure all components (index, docstore, model) are available and configured.")