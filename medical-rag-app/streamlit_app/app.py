# --- START OF FILE streamlit_app/app.py ---

import streamlit as st
import time
import sys
import os
from typing import List, Dict

# Add src directory to path for imports if running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.generator import MedicalRAGGenerator
    from src.retriever import MedicalRAGRetriever
    from streamlit_app.components import (
        display_ethical_disclaimer,
        create_feedback_section
    )
except ImportError as e:
    st.error(f"Error importing modules: {e}. Make sure the project structure is correct and all dependencies are installed.")
    st.stop()


def set_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Medical RAG AI Chat",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def sidebar_configuration() -> Dict:
    """
    Create sidebar for application configuration.

    Returns:
        Dict: Configuration settings
    """
    with st.sidebar:
        st.title("🔧 RAG Configuration")

        st.markdown("Adjust retrieval and generation settings:")

        config = {
            'num_contexts': st.slider(
                "Number of Context Sources",
                min_value=1, max_value=10, value=3,
                help="How many relevant text snippets to retrieve for context."
            ),
            'similarity_threshold': st.slider(
                "Context Relevance Threshold",
                min_value=0.0, max_value=1.0,
                value=0.5, step=0.05,
                help="Minimum similarity score for a context snippet to be considered relevant."
            ),
            # Removed 'response_length' as the LLM prompt encourages detailed responses
        }

        st.markdown("---")
        st.info("Higher 'Number of Contexts' and lower 'Relevance Threshold' might provide more information but could also introduce noise.")

        # You could add other settings here if needed
        # e.g., temperature, max_tokens for the LLM if you modify llm_interface

    return config

@st.cache_resource # Cache the loaded models for efficiency
def load_rag_components():
    """Loads the Retriever and Generator."""
    try:
        retriever = MedicalRAGRetriever()
        # Check if index exists before loading generator
        if retriever.index is None:
             st.error("FAISS index not found or failed to load. Please run indexing first (`python main.py --index`).")
             st.stop()
        generator = MedicalRAGGenerator(retriever)
        if generator.model is None:
             st.error("Medical LLM failed to load. Check model path and logs.")
             st.stop()
        return retriever, generator
    except Exception as e:
        st.error(f"Failed to initialize RAG components: {e}")
        st.stop()

def format_contexts_for_display(contexts: List[Dict]) -> str:
    """Formats the retrieved contexts into a markdown string for an expander."""
    if not contexts:
        return "No relevant contexts were found based on the current threshold."

    context_str = ""
    for i, ctx in enumerate(contexts):
        context_str += f"**Context {i+1} (Similarity: {ctx['similarity']:.3f})**\n"
        context_str += f"*   **Source:** `{ctx['document']}`\n"
        context_str += f"*   **Text:** {ctx['sentence']}\n\n"
    return context_str

def main():
    set_page_config()

    st.title("🩺 Medical RAG: AI Chat Assistant")
    st.caption("Ask medical questions and get insights backed by retrieved information.")

    # Display disclaimer prominently
    display_ethical_disclaimer()

    # Sidebar Configuration
    config = sidebar_configuration()

    # Load RAG components (cached)
    retriever, generator = load_rag_components()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm here to help with your medical questions. How can I assist you today? Remember, I am an AI and cannot replace a real doctor."}
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True) # Allow HTML for the expander later

    # Accept user input
    if user_query := st.chat_input("Ask your medical question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking... 🧠")
            try:
                start_time = time.time()

                # Generate response using RAG
                # Note: The generator internally uses the retriever passed during init
                response_text = generator.generate_response(
                    user_query,
                    num_contexts=config['num_contexts'],
                    similarity_threshold=config['similarity_threshold']
                    # The max_response_tokens is handled within generator/llm_interface
                )

                # Retrieve contexts *again* just for display purposes (or modify generator)
                # This is slightly redundant but keeps components separate
                contexts = retriever.retrieve_relevant_contexts(
                    user_query,
                    top_k=config['num_contexts'],
                    similarity_threshold=config['similarity_threshold']
                )
                end_time = time.time()

                # Format context display using <details>/<summary> HTML tags for a native-like expander
                formatted_contexts = format_contexts_for_display(contexts)
                context_expander = f"""
<details>
  <summary>📚 View Retrieved Contexts ({len(contexts)} found)</summary>
  <div style="padding: 10px; border: 1px solid #eee; margin-top: 5px; background-color: #f9f9f9;">
    {formatted_contexts.replace('`', '<code>').replace('`', '</code>')}
  </div>
</details>
"""

                # Combine response and context expander
                full_response = f"{response_text}\n\n{context_expander}"

                message_placeholder.markdown(full_response, unsafe_allow_html=True)
                st.caption(f"Response generated in {end_time - start_time:.2f} seconds.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                full_response = "Sorry, I encountered an error while processing your request. Please try again."
                message_placeholder.markdown(full_response)

            # Add assistant response (with context expander) to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Add feedback section at the end, but only if there's more than the initial message
    if len(st.session_state.get("messages", [])) > 1:
         create_feedback_section()


if __name__ == "__main__":
    # Ensure the script can find the 'src' directory when run directly
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    main()

# --- END OF FILE streamlit_app/app.py ---