import streamlit as st

def display_header():
    """Displays the application header."""
    # --- REMOVE set_page_config FROM HERE ---
    # st.set_page_config(page_title="Medical RAG Chatbot", layout="wide") # <-- DELETE OR COMMENT OUT THIS LINE
    # --- End of removed section ---
    st.title("🩺 Medical Document RAG Chatbot")
    st.caption("Ask questions about your indexed medical documents.")

def display_chat_message(role: str, content: str):
    """Displays a chat message."""
    with st.chat_message(role):
        st.markdown(content)

def get_user_query() -> str | None:
    """Gets the user query from the chat input."""
    if prompt := st.chat_input("Ask your question here..."):
        return prompt
    return None