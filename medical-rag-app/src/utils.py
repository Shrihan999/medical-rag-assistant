import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = ROOT_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"

# Model paths
MODELS_DIR = ROOT_DIR / "models"
LLM_MODEL_DIR = MODELS_DIR / "Med-Qwen2-7B-GGUF"

# Ensure directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
LLM_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Constants
# Choose an embedding model (adjust if needed)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Choose a GGUF file you download from the HF repo
# IMPORTANT: Update this filename based on the actual file you download
# Start with this recommended one
LLM_GGUF_FILE = "Med-Qwen2-7B.Q4_K_M.gguf"

# OR, if you want to try the higher quality one later:
# LLM_GGUF_FILE = "med-qwen2-7b.Q6_K.gguf"
LLM_MODEL_PATH = LLM_MODEL_DIR / LLM_GGUF_FILE

# Vector Store filename
INDEX_PATH = EMBEDDINGS_DIR / "faiss_index.index"
DOCSTORE_PATH = EMBEDDINGS_DIR / "docstore.pkl" # To store text chunks corresponding to index

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Retrieval parameters
TOP_K = 5 # Number of documents to retrieve

# LLM parameters (adjust based on your hardware)
LLM_N_CTX = 4096 # Context window size for the LLM
LLM_N_GPU_LAYERS = -1 # Set > 0 if you have llama-cpp-python built with GPU support (e.g., 30)
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1024

# Hugging Face Repo ID for downloading
LLM_REPO_ID = "mradermacher/Med-Qwen2-7B-GGUF"

print(f"--- Configuration ---")
print(f"Root Dir: {ROOT_DIR}")
print(f"Documents Dir: {DOCUMENTS_DIR}")
print(f"Embeddings Dir: {EMBEDDINGS_DIR}")
print(f"LLM Model Dir: {LLM_MODEL_DIR}")
print(f"LLM Model Path: {LLM_MODEL_PATH}")
print(f"FAISS Index Path: {INDEX_PATH}")
print(f"Docstore Path: {DOCSTORE_PATH}")
print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
print(f"---------------------")