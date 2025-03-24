import logging
import json
import os
from pathlib import Path
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

logger = logging.getLogger(__name__)

def download_model():
    """
    Download the medical LLM model from Hugging Face.
    
    Returns:
        str: Path to the downloaded model or None if an error occurred
    """
    try:
        model_id = "mradermacher/Med-Qwen2-7B-GGUF"  # Your model ID on Hugging Face
        
        logger.info(f"Downloading {model_id} model from Hugging Face...")
        
        # Download the model and tokenizer
        model = LlamaForCausalLM.from_pretrained(model_id)
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        
        # Save the model and tokenizer locally
        model_path = f"models/{model_id}"
        Path(model_path).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        logger.info(f"Model downloaded and saved to {model_path}")
        
        with open("models/model_info.json", "w") as f:
            json.dump({"model_id": model_id, "model_path": model_path}, f, indent=2)
        
        return model_path
    
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return None

def load_model():
    """
    Load the LLM model using transformers library.
    
    Returns:
        model: The loaded model instance or None if an error occurred
        tokenizer: The tokenizer instance
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
                return None, None
        
        # Load the model and tokenizer from the local directory
        model = LlamaForCausalLM.from_pretrained(model_path).to('cuda')  # Load on GPU
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        
        logger.info(f"Model loaded successfully from {model_path}")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

def generate_answer(model, tokenizer, question, context, max_tokens=512, temperature=0.1):
    """
    Generate an answer using the LLM.
    
    Args:
        model: The LLM model instance
        tokenizer: The tokenizer instance
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
        
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')  # Move input to GPU
        
        # Generate the answer
        logger.info("Generating answer...")
        output = model.generate(inputs['input_ids'], max_length=max_tokens, temperature=temperature)
        
        # Decode the output
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info("Answer generated successfully")
        
        return answer
    
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Error generating answer: {str(e)}"

# This is where you need to paste the corrected __main__ block
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])
    
    logger.info("Initializing model test...")
    model, tokenizer = load_model()
    if model and tokenizer:
        test_question = "What are common symptoms of diabetes?"
        test_context = "Diabetes symptoms include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, irritability, and blurred vision."
        
        # Pass both question and context to the generate_answer function
        answer = generate_answer(model, tokenizer, test_question, test_context)
        logger.info(f"Test answer: {answer}")
    else:
        logger.error("Failed to load model for testing.")
