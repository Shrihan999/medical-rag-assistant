# src/llm_interface.py

import logging
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files # Import list_repo_files for better error handling
from llama_cpp import Llama

# Import necessary configuration variables from src.utils
# Ensure these variables are correctly defined in your src/utils.py file
try:
    from src.utils import (
        LLM_REPO_ID,
        LLM_GGUF_FILE,
        LLM_MODEL_PATH,
        LLM_N_CTX,
        LLM_N_GPU_LAYERS,
        LLM_TEMPERATURE,
        LLM_MAX_TOKENS,
        LLM_MODEL_DIR
    )
except ImportError:
    logging.error("Could not import configuration from src.utils. Please ensure utils.py exists and contains the required variables.")
    # Define fallbacks or raise error if configuration is critical
    # For now, we'll let subsequent code fail if imports didn't work.
    pass


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable to hold the loaded LLM instance (cached)
llm_instance = None

def download_model():
    """Downloads the GGUF model file from Hugging Face Hub if it doesn't exist."""
    if LLM_MODEL_PATH.exists():
        logging.info(f"Model file already exists at {LLM_MODEL_PATH}. Skipping download.")
        return True

    logging.info(f"Downloading model {LLM_GGUF_FILE} from {LLM_REPO_ID} to {LLM_MODEL_DIR}...")
    try:
        hf_hub_download(
            repo_id=LLM_REPO_ID,
            filename=LLM_GGUF_FILE,
            local_dir=LLM_MODEL_DIR,
            local_dir_use_symlinks=False # Recommended for stability
        )
        # Verify download by checking file existence again
        if LLM_MODEL_PATH.exists():
             logging.info(f"Model downloaded successfully to {LLM_MODEL_PATH}")
             return True
        else:
             logging.error(f"Download command completed but model file still not found at {LLM_MODEL_PATH}. Check permissions or disk space.")
             return False
    except Exception as e:
        logging.error(f"Failed to download model {LLM_GGUF_FILE} from {LLM_REPO_ID}: {e}")
        # Attempt to list files in the repository to help the user find the correct name
        try:
            logging.info(f"Attempting to list files in repo {LLM_REPO_ID} to help diagnosis...")
            files = list_repo_files(LLM_REPO_ID)
            gguf_files = [f for f in files if f.endswith('.gguf')]
            if gguf_files:
                 logging.info(f"Available GGUF files in the repository: {gguf_files}")
                 logging.error(f"Please verify that the LLM_GGUF_FILE ('{LLM_GGUF_FILE}') in src/utils.py matches one of these available files *exactly*.")
            else:
                 logging.warning(f"No .gguf files found in the repository {LLM_REPO_ID}. Cannot download model.")
        except Exception as list_e:
            # Log error if listing files also fails
            logging.error(f"Could not list repository files either: {list_e}")
        return False

def load_llm() -> Llama | None:
    """
    Loads the GGUF model using llama-cpp-python.
    Returns the Llama instance or None if loading fails.
    Caches the loaded instance in the global 'llm_instance'.
    """
    global llm_instance
    if llm_instance is None:
        if not LLM_MODEL_PATH.exists():
            logging.error(f"Model file not found at {LLM_MODEL_PATH}. Cannot load LLM.")
            logging.error("Please download the model first (e.g., using 'python main.py --download-model').")
            # Optional: raise FileNotFoundError if you want loading failure to halt execution
            # raise FileNotFoundError(f"Model file not found: {LLM_MODEL_PATH}")
            return None # Return None if loading is not possible

        logging.info(f"Loading LLM from {LLM_MODEL_PATH}...")
        try:
            # Log parameters being used for loading
            logging.info(f"Initializing Llama with: n_ctx={LLM_N_CTX}, n_gpu_layers={LLM_N_GPU_LAYERS}")
            llm_instance = Llama(
                model_path=str(LLM_MODEL_PATH),
                n_ctx=LLM_N_CTX,                # Context window size
                n_gpu_layers=LLM_N_GPU_LAYERS,  # Number of layers to offload to GPU (0 for CPU only, -1 for max possible)
                n_batch=512,                    # Batch size for prompt processing (adjust based on RAM/VRAM)
                verbose=False                   # Set to True for detailed Llama.cpp logs during init/generation
            )
            logging.info("LLM loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load LLM from {LLM_MODEL_PATH}: {e}")
            # Provide specific feedback for common errors
            if "Could not load model" in str(e):
                 logging.error("Possible reasons: Incorrect model file path, corrupted download, or llama-cpp-python build issues.")
            elif "failed to allocate" in str(e):
                 logging.error("Memory allocation error. Possible reasons: Insufficient RAM/VRAM. Try reducing n_ctx, n_batch, or using a smaller model quantization.")
            elif "cuda" in str(e).lower() or "cublas" in str(e).lower() or "gpu" in str(e).lower() and LLM_N_GPU_LAYERS != 0:
                 logging.error(f"GPU-related error during LLM loading. Ensure llama-cpp-python was built correctly with CUDA support (using CMAKE_ARGS='-DGGML_CUDA=on' for pip install) and that n_gpu_layers ({LLM_N_GPU_LAYERS}) is valid for your setup.")
            else:
                 logging.error("An unexpected error occurred during model loading.")

            llm_instance = None # Ensure instance remains None if loading failed
            # Optional: Re-raise the exception if you want loading failure to stop the application
            # raise e
    return llm_instance

def create_prompt(query: str, context: str) -> str:
    """
    Creates a detailed, structured prompt for the LLM, instructing it to
    base its response strictly on the provided context.
    """
    # This prompt structure requests detailed sections but emphasizes using ONLY the provided context.
    prompt = f"""You are an expert medical AI assistant. Your task is to answer the following medical query based *solely* on the provided context documents.

Instructions:
1.  Carefully read the provided context below.
2.  Generate a detailed, multi-section response to the query.
3.  Structure your response covering the following points *if the information is present in the context*:
    - Comprehensive medical explanation related to the query.
    - Potential causes mentioned in the context.
    - Diagnostic considerations discussed in the context.
    - Treatment options described in the context.
    - Preventive measures outlined in the context.
    - Additional relevant medical insights found *only* in the context.
4.  **Crucially: If the context does not contain information for a specific section, explicitly state "Information on [Section Name] is not available in the provided documents." Do NOT invent information or use external knowledge.**
5.  Ensure the response is accurate according to the context, comprehensive where possible based on the context, and clearly structured. Use formatting like lists or bullet points if it enhances clarity and is supported by the context.

Provided Context:
--- START CONTEXT ---
{context}
--- END CONTEXT ---

Medical Query: {query}

Detailed Medical Response (Based *only* on provided context):
"""
    return prompt

def generate_response(query: str, context: str) -> str:
    """
    Generates a response from the LLM given a query and context.
    Handles loading the LLM instance and potential errors during generation.
    """
    llm = load_llm() # Attempt to get or load the LLM instance
    if llm is None:
        # If LLM couldn't be loaded (e.g., file missing, load error), return an error message.
        return "Error: The Language Model could not be loaded. Please check logs and ensure the model is downloaded and configured correctly."

    # Create the detailed prompt using the dedicated function
    prompt = create_prompt(query, context)
    logging.info(f"Using detailed prompt (approx. length: {len(prompt)} chars)") # Log prompt length for debugging

    logging.info(f"Generating response for query: '{query[:100]}...'") # Log start of generation
    try:
        # Call the Llama instance to generate text
        output = llm(
            prompt,
            max_tokens=LLM_MAX_TOKENS,       # Max new tokens to generate (set high in utils.py, e.g., 1024+)
            temperature=LLM_TEMPERATURE,    # Controls randomness (set in utils.py)
            # Stop sequences help prevent the model from generating past the intended answer
            # or repeating parts of the prompt.
            stop=["Medical Query:", "Question:", "--- END CONTEXT ---", "--- START CONTEXT ---", "\n\nHuman:", "\n\nAssistant:"],
            echo=False                      # False = Don't include the prompt in the output string
        )

        # Extract the generated text and clean up any leading/trailing whitespace
        response = output['choices'][0]['text'].strip()

        logging.info(f"Response generated successfully (length: {len(response)} chars).")
        return response

    except Exception as e:
        # Log errors specifically occurring during the generation step
        logging.error(f"Error during LLM generation for query '{query[:100]}...': {e}")
        # Check if the error might be related to exceeding the context window size
        if "n_ctx" in str(e).lower() or "context window" in str(e).lower():
             logging.error(f"Context window potentially exceeded during generation. Prompt length: {len(prompt)}, LLM context size (n_ctx): {LLM_N_CTX}.")
             logging.error("Consider reducing TOP_K in utils.py, shortening the prompt, or using an LLM with a larger context window if available.")
             return "Error: The request was too long for the model's context window. Please try a shorter query or adjust settings."
        else:
             # Generic error message for other generation failures
             return "Error: An unexpected error occurred while generating the response from the LLM."


# Standalone execution block for testing this module directly
if __name__ == '__main__':
    print("--- Running llm_interface.py Standalone Test ---")

    print("\nStep 1: Attempting to download model (if missing)...")
    if download_model(): # Proceed only if model is available or downloaded
        print("Model is available or downloaded successfully.")

        print("\nStep 2: Attempting to load LLM...")
        try:
            # Attempt to load the LLM instance
            llm_instance_test = load_llm()
            if llm_instance_test:
                print("LLM loaded successfully for testing.")

                # Define a test query and some sample context
                test_query = "What are common treatments for hypertension based on this text?"
                test_context = """Hypertension, or high blood pressure, is a common condition. Lifestyle changes are often the first line of treatment, including adopting a healthy diet (like the DASH diet), reducing sodium intake, regular physical activity, maintaining a healthy weight, and limiting alcohol consumption. If lifestyle changes aren't enough, medications may be prescribed. Common classes of antihypertensive medications include diuretics, ACE inhibitors, ARBs, beta-blockers, and calcium channel blockers. The choice of medication depends on the individual's blood pressure level, overall health, and presence of other medical conditions. Diagnostic considerations involve multiple blood pressure readings. Preventive measures focus on healthy lifestyle choices from an early age."""

                print(f"\nTest Query: {test_query}")
                print("\nStep 3: Generating response using detailed prompt...")

                # Generate response using the function from this module
                response = generate_response(test_query, test_context)

                print("\n--- Generated Response ---")
                print(response)
                print("--------------------------")

            else:
                # Handle case where load_llm returned None
                print("LLM loading failed. Cannot proceed with generation test.")
                print("Please check previous error messages for details (e.g., model path, build issues, memory).")

        except Exception as e:
            # Catch any other exceptions during the loading/generation test
            print(f"\nAn error occurred during the standalone test: {e}")
            logging.exception("Standalone test failed.") # Log the full traceback for debugging
            print("Ensure the model file exists, paths are correct, and llama-cpp-python is correctly installed (potentially with GPU support if desired).")
    else:
        # Handle case where download_model returned False
        print("Model download failed or model is unavailable. Cannot proceed with LLM tests.")
        print(f"Please ensure the model file '{LLM_GGUF_FILE}' can be downloaded from '{LLM_REPO_ID}' or exists at '{LLM_MODEL_PATH}'.")

    print("\n--- Standalone Test Finished ---")