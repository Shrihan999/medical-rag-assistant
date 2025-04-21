import streamlit as st # Make sure streamlit is imported
import sys
import os
from pathlib import Path
import logging

# --- MOVE set_page_config HERE ---
# This MUST be the first Streamlit command
st.set_page_config(page_title="Medical RAG Chatbot", layout="wide")
# --- End of moved section ---

# Ensure the src directory is in the Python path
# This is crucial for Streamlit to find your modules
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
# print("sys.path:", sys.path) # Uncomment for debugging path issues

# Set environment variable to potentially mitigate threading issues with tokenizers in Hugging Face
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Import components after adjusting path
    from src.generator import RAGGenerator
    from src.utils import INDEX_PATH, DOCSTORE_PATH, LLM_MODEL_PATH
    # Import display_header even though set_page_config moved
    from streamlit_app.components import display_header, display_chat_message, get_user_query
except ImportError as e:
    st.error(f"Failed to import necessary modules. Please ensure all files are correctly placed and requirements installed. Error: {e}")
    st.stop() # Stop execution if imports fail


# --- Initialization ---
@st.cache_resource # Cache the generator instance for efficiency
def load_rag_generator():
    """Loads the RAG Generator instance, handling potential errors."""
    # It's okay for st.info/success etc. to be here now
    st.info("Initializing RAG system... (This might take a moment on first run)")
    try:
        # Check if necessary files exist before initializing
        if not INDEX_PATH.exists() or not DOCSTORE_PATH.exists():
            st.warning(f"Index file ({INDEX_PATH.name}) or docstore ({DOCSTORE_PATH.name}) not found in {INDEX_PATH.parent}. Please run indexing first (e.g., `python main.py --index`). Retrieval will be disabled.")
            # Optionally, you could prevent the app from running entirely
            # st.error("Cannot start: Index files missing.")
            # st.stop()

        if not LLM_MODEL_PATH.exists():
             st.warning(f"LLM model file ({LLM_MODEL_PATH.name}) not found in {LLM_MODEL_PATH.parent}. Please download the model first (e.g., `python main.py --download-model`). Generation will fail.")
             # Optionally, stop here too
             # st.error("Cannot start: LLM model missing.")
             # st.stop()

        generator = RAGGenerator()
        st.success("RAG system initialized successfully!")
        return generator
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}. Please check logs and setup.")
        # Log the full traceback for debugging
        logging.exception("RAG Generator initialization failed in Streamlit app.")
        return None

# Call load_rag_generator AFTER set_page_config
rag_generator = load_rag_generator()

# --- Streamlit UI ---
# display_header can now be called safely
display_header()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    display_chat_message(message["role"], message["content"])

# Get user input
user_query = get_user_query()

if user_query:
    # Display user message
    display_chat_message("user", user_query)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    if rag_generator is None:
        error_message = "RAG system is not available due to initialization errors."
        display_chat_message("assistant", error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})
    else:
        # Generate response
        with st.spinner("Thinking..."):
            try:
                response = rag_generator.answer_query(user_query)
                # Display assistant response
                display_chat_message("assistant", response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                 error_message = f"An error occurred while generating the response: {e}"
                 logging.exception(f"Error processing query '{user_query}' in Streamlit app.")
                 display_chat_message("assistant", error_message)
                 st.session_state.messages.append({"role": "assistant", "content": error_message})

# Add a sidebar note about potential first-time loading slowness
st.sidebar.info("Note: The first query might take longer as the model loads into memory.")
st.sidebar.markdown("---")
st.sidebar.caption(f"Using model: `{LLM_MODEL_PATH.name}`")
st.sidebar.caption(f"Index: `{INDEX_PATH.name}`")