"""
LLM interface module for the Medical RAG application.
Handles interaction with the LLM model.
"""

import logging
import json
import os
from pathlib import Path


logger = logging.getLogger(__name__)

def download_model():
    """
    Download the medical LLM model from Hugging Face.
    
    Returns:
        str: Path to the downloaded model or None if an error occurred
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        
        logger.info("Downloading Med-Qwen2-7B-GGUF model from Hugging Face...")
        model_id = "mradermacher/Med-Qwen2-7B-GGUF"
        
        logger.info("Checking model files...")
        try:
            files = list_repo_files(model_id)
            gguf_files = [f for f in files if f.endswith(".gguf")]
            model_filename = gguf_files[0] if gguf_files else "Med-Qwen2-7b.Q4_K_M.gguf"
            logger.info(f"Found model file: {model_filename}")
        except Exception as e:
            logger.warning(f"Could not list repo files: {e}")
            model_filename = "Med-Qwen2-7b.Q4_K_M.gguf"
        
        output_path = "models/Med-Qwen2-7B-GGUF"
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        model_path = hf_hub_download(
            repo_id=model_id,
            filename=model_filename,
            local_dir=output_path,
        )
        
        logger.info(f"Model downloaded and saved to {model_path}")
        
        with open("models/model_info.json", "w") as f:
            json.dump({"model_id": model_id, "model_filename": model_filename, "model_path": model_path}, f, indent=2)
        
        return model_path
    
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return None

def load_model():
    """
    Load the LLM model.
    
    Returns:
        LLM: The loaded model instance or None if an error occurred
    """
    try:
        if not os.path.exists("models/model_info.json"):
            logger.warning("Model info not found. Downloading model first...")
            download_model()
        
        with open("models/model_info.json", "r") as f:
            model_info = json.load(f)
        
        model_path = model_info["model_path"]
        
        if not os.path.exists(model_path):
            logger.warning("Model file not found. Downloading model...")
            model_path = download_model()
            if model_path is None:
                return None
        
        from llama_cpp import Llama
        
        logger.info(f"Loading model from {model_path}...")
        model = Llama(model_path=model_path, n_ctx=2048, n_threads=4)
        logger.info("Model loaded successfully")
        return model
    
    except ImportError:
        logger.error("llama-cpp-python not installed. Please install it with: pip install llama-cpp-python")
        return None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def generate_answer(model, question, context, max_tokens=512, temperature=0.1):
    """
    Generate an answer using the LLM.
    
    Args:
        model: The LLM model instance
        question (str): The user's question
        context (str): Retrieved context from the RAG system
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature
        
    Returns:
        str: The generated answer
    """
    try:
        if model is None:
            logger.error("Model not loaded. Cannot generate answer.")
            return "Error: Model not loaded. Please load the model first."
        
        prompt = f"""
You are a helpful AI medical assistant. Answer the following medical question using ONLY the given context information.
If the context doesn't contain relevant information, say "I don't have enough information to answer this question."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
        
        logger.info("Generating answer...")
        response = model.generate(prompt, max_tokens=max_tokens, temperature=temperature, stop=["QUESTION:", "\n\n"])
        
        answer = response.strip()
        logger.info("Answer generated successfully")
        
        return answer
    
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Error generating answer: {str(e)}"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])
    
    logger.info("Initializing model test...")
    model = load_model()
    if model:
        test_question = "What are common symptoms of diabetes?"
        test_context = "Diabetes symptoms include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, irritability, and blurred vision."
        
        answer = generate_answer(model, test_question, test_context)
        logger.info(f"Test answer: {answer}")
    else:
        logger.error("Failed to load model for testing.")
