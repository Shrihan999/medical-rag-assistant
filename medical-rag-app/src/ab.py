import os
import logging
import torch
from llama_cpp import Llama

def detailed_gpu_diagnostics():
    """
    Comprehensive GPU and model loading diagnostics.
    """
    print("=== System and CUDA Diagnostics ===")
    print(f"Torch Version: {torch.__version__}")
    print(f"Llama-CPP Version: {__import__('llama_cpp').__version__}")
    
    # CUDA Diagnostics
    print("\n=== CUDA Information ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    
    # Find GGUF model
    model_dir = "models/Med-Qwen2-7B-GGUF"
    gguf_files = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]
    
    if not gguf_files:
        print("\n!!! ERROR: No GGUF files found in the model directory !!!")
        return
    
    model_path = os.path.join(model_dir, gguf_files[0])
    print(f"\n=== Model File Diagnostics ===")
    print(f"Model Path: {model_path}")
    print(f"Model File Exists: {os.path.exists(model_path)}")
    print(f"Model File Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    # Attempt model loading with detailed logging
    print("\n=== Model Loading Attempt ===")
    try:
        model = Llama(
            model_path=model_path,
            n_ctx=4096,  # Context window
            n_batch=512,  # Batch size
            n_gpu_layers=-1,  # Attempt to use all GPU layers
            main_gpu=0,  # Use first GPU
            verbose=True  # Detailed logging
        )
        print("Model loaded successfully!")
        
        # Test generation
        print("\n=== Generation Test ===")
        response = model.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain GPU acceleration in machine learning."}
            ],
            max_tokens=100
        )
        print("Generation completed successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

# Run diagnostics
detailed_gpu_diagnostics()