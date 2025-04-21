import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Ensure the src directory is in the Python path for module imports
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))
# print("sys.path:", sys.path) # Uncomment for debugging path issues

try:
    from src.data_processor import process_documents
    from src.indexer import build_index
    from src.llm_interface import download_model
    from src.utils import LLM_MODEL_PATH, INDEX_PATH, DOCSTORE_PATH # Import paths for checks
except ImportError as e:
    print(f"Error: Failed to import necessary modules from 'src'. Make sure 'src' is in the Python path and all files exist. Details: {e}")
    sys.exit(1) # Exit if core modules can't be imported

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_streamlit():
    """Launches the Streamlit application."""
    streamlit_app_path = root_dir / "streamlit_app" / "app.py"
    if not streamlit_app_path.exists():
        logging.error(f"Streamlit app file not found at {streamlit_app_path}")
        return

    # Check for necessary files before launching Streamlit
    if not INDEX_PATH.exists() or not DOCSTORE_PATH.exists():
        logging.warning(f"Index file ({INDEX_PATH.name}) or docstore ({DOCSTORE_PATH.name}) not found. Streamlit app might not function correctly for retrieval. Consider running --index.")

    if not LLM_MODEL_PATH.exists():
        logging.warning(f"LLM model file ({LLM_MODEL_PATH.name}) not found. Streamlit app might fail during generation. Consider running --download-model.")


    logging.info(f"Launching Streamlit app from {streamlit_app_path}...")
    try:
        # Use sys.executable to ensure streamlit runs with the same python env
        # Use Popen for potentially better control/less blocking if needed, but run is simpler.
        # Use shell=True carefully, or construct command list: [sys.executable, "-m", "streamlit", "run", str(streamlit_app_path)]
        # For simplicity with potential path spaces, shell=True might be easier here.
        # IMPORTANT: Ensure 'streamlit' command is available in the environment.
        command = f"{sys.executable} -m streamlit run {streamlit_app_path}"
        subprocess.run(command, shell=True, check=True) # Add check=True to raise error if streamlit fails
    except FileNotFoundError:
        logging.error("Error: 'streamlit' command not found. Please ensure Streamlit is installed in your environment (`pip install streamlit`).")
    except subprocess.CalledProcessError as e:
         logging.error(f"Streamlit app failed to run: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while trying to launch Streamlit: {e}")

def main():
    parser = argparse.ArgumentParser(description="Medical RAG Application Pipeline")
    parser.add_argument("--preprocess", action="store_true", help="Process PDF documents in data/documents")
    parser.add_argument("--index", action="store_true", help="Generate embeddings and create vector index for processed PDFs")
    parser.add_argument("--download-model", action="store_true", help="Download the medical LLM model (GGUF)")
    parser.add_argument("--streamlit", action="store_true", help="Launch the Streamlit web interface")
    parser.add_argument("--all", action="store_true", help="Perform all steps: download model, preprocess, index")

    args = parser.parse_args()

    # Determine execution order for --all
    run_download = args.download_model or args.all
    run_preprocess = args.preprocess or args.all
    run_index = args.index or args.all
    run_app = args.streamlit

    # --- Execute Steps ---
    if run_download:
        logging.info("--- Running Model Download ---")
        if not download_model():
            logging.error("Model download failed. Subsequent steps might fail.")
            # Optionally exit if model is critical for next steps like indexing (if using model-based chunking)
            # sys.exit(1)
        logging.info("--- Model Download Finished ---")

    processed_chunks_for_indexing = None
    if run_preprocess:
        logging.info("--- Running Document Preprocessing ---")
        processed_chunks_for_indexing = process_documents()
        if not processed_chunks_for_indexing:
             logging.warning("Preprocessing resulted in no chunks. Indexing step will be skipped if run.")
        logging.info("--- Document Preprocessing Finished ---")

    if run_index:
        logging.info("--- Running Indexing ---")
        # If --index is run alone, we need to run preprocessing first
        if processed_chunks_for_indexing is None and not args.preprocess and not args.all:
            logging.info("Preprocessing needed before indexing. Running preprocessing first...")
            processed_chunks_for_indexing = process_documents()

        if processed_chunks_for_indexing:
            build_index(processed_chunks_for_indexing)
        else:
            logging.error("Cannot run indexing because no processed chunks are available. Run --preprocess first or check the data/documents folder.")
        logging.info("--- Indexing Finished ---")

    if run_app:
        logging.info("--- Launching Streamlit App ---")
        run_streamlit()
        logging.info("--- Streamlit App Closed ---")

    # If no arguments were given, print help
    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main()