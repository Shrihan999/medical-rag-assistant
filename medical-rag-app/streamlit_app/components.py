# --- START OF FILE streamlit_app/components.py ---

import streamlit as st
from typing import List, Dict

# Keep this function as is
def display_ethical_disclaimer():
    """
    Display an ethical disclaimer for medical AI advice.
    """
    st.markdown("""
    <div style='background-color: #FFF3CD; color: #856404; border-left: 6px solid #FFC107; padding: 10px; margin-bottom: 15px;'>
        <h4 style='margin-top: 0; color: #856404;'>⚠️ Important Disclaimer</h4>
        <p>This is an AI-powered assistant and should <strong>NOT</strong> replace professional medical advice.</p>
        <ul>
            <li>Consult a licensed healthcare professional for personalized medical guidance.</li>
            <li>AI responses are based on available information and may not be fully comprehensive or accurate for your specific situation.</li>
            <li>Individual medical conditions require expert human evaluation.</li>
            <li>Always seek direct medical consultation for health concerns.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# Keep this function as is
def create_feedback_section():
    """
    Create a user feedback section for the medical AI.
    """
    st.markdown("---") # Add a divider
    st.subheader("Feedback on the Last Response")

    feedback_key_base = f"feedback_{len(st.session_state.get('messages', []))}" # Unique key per response

    helpful = st.radio(
        "Was this response helpful?",
        ["Select", "Yes", "No"],
        key=f"{feedback_key_base}_helpful"
    )

    comment = st.text_area(
        "Additional comments (optional)",
        key=f"{feedback_key_base}_comment"
    )

    if st.button("Submit Feedback", key=f"{feedback_key_base}_submit"):
        if helpful != "Select":
            # Here you would ideally log the feedback (helpful status, comment, query, response)
            st.success("Thank you for your feedback!")
            # You might want to disable the feedback widgets after submission
        else:
            st.warning("Please select 'Yes' or 'No'.")


# Remove display_context_sources, create_similarity_chart, highlight_key_insights
# These functionalities will be integrated differently or removed.

# --- END OF FILE streamlit_app/components.py ---