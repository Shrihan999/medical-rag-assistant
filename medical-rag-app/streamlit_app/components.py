"""
Components for the Streamlit Medical RAG application.
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

def render_header():
    """Render the application header."""
    st.title("Medical RAG Assistant")
    st.markdown("""
    This application uses a medical language model with Retrieval Augmented Generation (RAG)
    to answer your medical questions based on a medical QA dataset.
    """)
    st.markdown("---")

def render_sidebar():
    """Render the sidebar with settings and info."""
    st.sidebar.title("Settings")
    
    top_k = st.sidebar.slider(
        "Number of documents to retrieve", 
        min_value=1, 
        max_value=10, 
        value=5,
        help="Higher values may provide more information but can introduce noise."
    )
    
    temperature = st.sidebar.slider(
        "Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1, 
        step=0.1,
        help="Higher values make output more random, lower values more deterministic."
    )
    
    max_tokens = st.sidebar.slider(
        "Maximum tokens in response", 
        min_value=64, 
        max_value=1024, 
        value=512, 
        step=64,
        help="Maximum number of tokens to generate for the answer."
    )
    
    show_retrieved_docs = st.sidebar.checkbox(
        "Show retrieved documents", 
        value=True,
        help="Display the documents retrieved for the question."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    This application uses:
    - Medical QA Dataset: [Malikeh1375/medical-question-answering-datasets](https://huggingface.co/datasets/Malikeh1375/medical-question-answering-datasets)
    - LLM Model: [mradermacher/Med-Qwen2-7B-GGUF](https://huggingface.co/mradermacher/Med-Qwen2-7B-GGUF)
    - Sentence Transformers for embeddings
    - FAISS for vector search
    """)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Check System Status"):
        system_status_tab()
    
    return {
        "top_k": top_k,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "show_retrieved_docs": show_retrieved_docs
    }

def render_question_input():
    """Render the question input area."""
    sample_questions = [
        "What are the symptoms of diabetes?",
        "How is pneumonia diagnosed?",
        "What are the side effects of ibuprofen?",
        "What is the treatment for hypertension?",
        "How does COVID-19 spread?"
    ]
    
    st.markdown("### Ask a Medical Question")
    
    # Sample questions
    st.markdown("#### Sample Questions:")
    cols = st.columns(len(sample_questions))
    
    # Init session state for question if not exists
    if "question" not in st.session_state:
        st.session_state.question = ""
    
    # Add sample question buttons
    for i, col in enumerate(cols):
        if col.button(f"Sample {i+1}", key=f"sample_{i}"):
            st.session_state.question = sample_questions[i]
    
    # Question input
    question = st.text_area(
        "Enter your medical question:",
        height=100,
        key="question"
    )
    
    return question

def render_answer(answer, retrieved_docs, show_retrieved_docs):
    """Render the answer and retrieved documents."""
    st.markdown("### Answer")
    st.markdown(answer)
    
    if show_retrieved_docs and retrieved_docs:
        st.markdown("---")
        st.markdown("### Retrieved Documents")
        
        for i, doc in enumerate(retrieved_docs):
            with st.expander(f"Document {i+1} (Relevance Score: {doc['score']:.4f})"):
                st.markdown(f"**Question:**\n{doc['metadata']['question']}")
                st.markdown(f"**Answer:**\n{doc['metadata']['answer']}")

def system_status_tab():
    """Display system status information."""
    st.sidebar.markdown("### System Status")
    
    # Check file status
    from src.utils import check_file_status, check_requirements
    
    files = check_file_status()
    requirements = check_requirements()
    
    # Display requirements status
    st.sidebar.markdown("#### Dependencies")
    req_status = pd.DataFrame({
        "Dependency": list(requirements.keys()),
        "Status": ["Installed" if status else "Missing" for status in requirements.values()]
    })
    st.sidebar.dataframe(req_status, hide_index=True)
    
    # Display file status
    st.sidebar.markdown("#### Data Files")
    file_status = []
    for file_name, file_info in files.items():
        if file_name != "model_file":  # Handle model file separately
            file_status.append({
                "File": file_name,
                "Status": "✅ Ready" if file_info["exists"] else "❌ Missing"
            })
    
    st.sidebar.dataframe(pd.DataFrame(file_status), hide_index=True)
    
    # Model status
    st.sidebar.markdown("#### Model Status")
    model_status = "✅ Downloaded" if files["model_file"]["exists"] else "❌ Not downloaded"
    st.sidebar.markdown(f"Model: {model_status}")
    
    # Recommendations
    st.sidebar.markdown("#### Recommendations")
    if not all(requirements.values()):
        st.sidebar.warning("Some dependencies are missing. Run `pip install -r requirements.txt` to install them.")
    
    if not files["dataset"]["exists"]:
        st.sidebar.warning("Dataset not found. Run `python main.py --preprocess` to download it.")
    
    if not files["processed_documents"]["exists"]:
        st.sidebar.warning("Processed documents not found. Run `python main.py --preprocess` to create them.")
    
    if not files["embeddings"]["exists"] or not files["faiss_index"]["exists"]:
        st.sidebar.warning("Embeddings or index not found. Run `python main.py --index` to create them.")
    
    if not files["model_file"]["exists"]:
        st.sidebar.warning("Model not found. Run `python main.py --download-model` to download it.")

def render_history():
    """Render the conversation history."""
    if "history" not in st.session_state:
        st.session_state.history = []
    
    if st.session_state.history:
        st.markdown("### Conversation History")
        
        for i, exchange in enumerate(st.session_state.history):
            st.markdown(f"**Question {i+1}:**")
            st.markdown(exchange["question"])
            st.markdown(f"**Answer {i+1}:**")
            st.markdown(exchange["answer"])
            st.markdown("---")