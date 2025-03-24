"""
Main Streamlit application for Medical RAG.
"""

import streamlit as st
import logging
import os
from pathlib import Path
import sys

# Add the root directory to the Python path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.generator import Generator
from src.utils import log_query, create_directories, check_file_status
from streamlit_app.components import (
    render_header, 
    render_sidebar, 
    render_question_input, 
    render_answer,
    render_history
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("streamlit_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def initialize_app():
    """Initialize the application."""
    # Create directories
    create_directories()
    
    # Check file status
    files = check_file_status()
    
    # Initialize session state
    if "generator" not in st.session_state:
        st.session_state.generator = Generator()
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Check if data is ready
    data_ready = files["processed_documents"]["exists"] and files["faiss_index"]["exists"]
    model_ready = files["model_file"]["exists"]
    
    return data_ready, model_ready

def main():
    """Main function for the Streamlit app."""
    st.set_page_config(
        page_title="Medical RAG Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize app
    data_ready, model_ready = initialize_app()
    
    # Render UI components
    render_header()
    settings = render_sidebar()
    
    # Check if data is ready
    if not data_ready:
        st.error("""
        Data not ready! Please run the following commands in the terminal:
        ```
        python main.py --preprocess
        python main.py --index
        ```
        """)
        return
    
    # Display warning if model is not ready
    if not model_ready:
        st.warning("""
        Model not downloaded! Please run the following command in the terminal:
        ```
        python main.py --download-model
        ```
        
        You can still ask questions, but the application will attempt to download
        the model first, which may take some time.
        """)
    
    # Get user question
    question = render_question_input()
    
    # Process question
    if st.button("Get Answer") and question:
        with st.spinner("Retrieving and generating answer..."):
            try:
                # Generate answer
                result = st.session_state.generator.answer_question(
                    question,
                    top_k=settings["top_k"],
                    max_tokens=settings["max_tokens"],
                    temperature=settings["temperature"]
                )
                
                answer = result["answer"]
                retrieved_docs = result.get("documents", [])
                
                # Add to history
                st.session_state.history.append({
                    "question": question,
                    "answer": answer,
                    "documents": retrieved_docs
                })
                
                # Log query
                log_query(question, answer, retrieved_docs)
                
                # Clear the question input
                if question:
                    # Instead of modifying session state directly, use a callback function
                    def clear_text():
                        st.session_state["question"] = ""
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                answer = f"Error: {str(e)}"
                retrieved_docs = []
        
        # Render answer and retrieved documents
        render_answer(answer, retrieved_docs, settings["show_retrieved_docs"])
    
    # Render history
    render_history()

if __name__ == "__main__":
    main()