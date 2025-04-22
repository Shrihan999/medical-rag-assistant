import os
import logging
import torch
from llama_cpp import Llama
import requests

logger = logging.getLogger(__name__)

def download_model(
    model_url="https://huggingface.co/mradermacher/Med-Qwen2-7B-GGUF/resolve/main/Med-Qwen2-7B.Q4_K_S.gguf", 
    model_dir="models/Med-Qwen2-7B-GGUF"
):
    """Download the specific GGUF model for medical RAG."""
    os.makedirs(model_dir, exist_ok=True)
    
    filename = os.path.basename(model_url)
    model_path = os.path.join(model_dir, filename)
    
    if os.path.exists(model_path):
        logger.info(f"Model {filename} already exists.")
        return model_path
    
    try:
        logger.info(f"Downloading model from {model_url}")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Model downloaded successfully to {model_path}")
        return model_path
    
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return None

def _determine_gpu_layers():
    """Dynamically determine the number of GPU layers based on available GPU memory."""
    try:
        if not torch.cuda.is_available():
            return 0
        
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        if total_memory > 16 * 1024 * 1024 * 1024:  # > 16GB
            return 50
        elif total_memory > 8 * 1024 * 1024 * 1024:  # 8-16GB
            return 32
        else:
            return 16
    
    except Exception as e:
        logger.warning(f"Unable to determine GPU layers: {e}")
        return 0

def load_medical_llm(
    model_path: str = None, 
    max_tokens: int = 1024,
    context_length: int = 4096
):
    """
    Load the medical LLM model using llama-cpp-python with GPU support.
    
    Args:
        model_path (str): Path to the GGUF model file
        max_tokens (int): Maximum number of tokens to generate
        context_length (int): Context window size
    
    Returns:
        Callable LLM model or None
    """
    if not model_path:
        model_dir = "models/Med-Qwen2-7B-GGUF"
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]
        
        if not model_files:
            logger.error("No GGUF model files found.")
            return None
        
        preferred_model = "Med-Qwen2-7B.Q4_K_S.gguf"
        model_path = os.path.join(
            model_dir, 
            preferred_model if preferred_model in model_files else model_files[0]
        )
    
    try:
        n_gpu_layers = _determine_gpu_layers()
        
        logger.info(f"Loading model: {model_path}")
        logger.info(f"Using {n_gpu_layers} GPU layers")
        
        model = Llama(
            model_path=model_path,
            n_ctx=context_length,
            n_batch=512,
            n_gpu_layers=n_gpu_layers,
            seed=-1,
            verbose=False
        )
        
        def model_call(prompt, max_new_tokens=max_tokens):
            """
            Generate a detailed medical response using the LLM.
            
            Args:
                prompt (str): Input medical query
                max_new_tokens (int): Maximum tokens to generate
            
            Returns:
                str: Comprehensive medical response
            """
            try:
                # Create a series of prompts to encourage detailed response
                detailed_prompts = [
                    f"""You are an expert medical professional providing a comprehensive, 
                    in-depth explanation to the following medical query. 
                    Provide a detailed, multi-section response that covers:
                    - Comprehensive medical explanation
                    - Potential causes
                    - Diagnostic considerations
                    - Treatment options
                    - Preventive measures
                    - Additional medical insights

                    Medical Query: {prompt}

                    Detailed Medical Response:""",
                    
                    f"""Please provide a thorough, academic-level medical explanation 
                    for the following query. Ensure your response is:
                    1. Scientifically accurate
                    2. Comprehensive
                    3. Structured with clear sections
                    4. Providing depth of medical knowledge
                    5. Use lists and bullets if needed and make sure to present the response in a nice formatting

                    Query: {prompt}

                    Expert Medical Analysis:"""
                ]
                
                # Try multiple prompts to get a detailed response
                for detailed_prompt in detailed_prompts:
                    response = model(
                        detailed_prompt, 
                        max_tokens=max_new_tokens,
                        stop=[],
                        echo=False,
                        temperature=0.7,
                        top_p=0.95,
                        top_k=50,
                        repeat_penalty=1.2
                    )
                    
                    generated_text = response.get('choices', [{}])[0].get('text', '').strip()
                    
                    # Check if response is meaningful
                    if len(generated_text) > 200:
                        return generated_text
                
                # Fallback if no detailed response generated
                return (
                    "I apologize, but I'm unable to generate a comprehensive medical response. "
                    "The query may be too complex or specific. "
                    "For precise medical advice, please consult a healthcare professional."
                )
            
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                return (
                    "I apologize, but I'm unable to generate a comprehensive response. "
                    "This could be due to technical limitations. "
                    "For precise medical advice, please consult a healthcare professional."
                )

        logger.info("Medical LLM loaded successfully with enhanced response generation")
        return model_call
    
    except Exception as e:
        logger.error(f"Comprehensive error loading medical LLM: {e}")
        return None

# Optional model download and loading
if __name__ == "__main__":
    model_path = download_model()
    if model_path:
        llm = load_medical_llm(model_path)
        if llm:
            print("Model loaded successfully!")