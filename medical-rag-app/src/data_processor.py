import logging
from pathlib import Path
from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.utils import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts text from a single PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:  # Ensure text was extracted
                text += page_text + "\n" # Add newline between pages
        logging.info(f"Successfully extracted text from {pdf_path.name}")
        return text
    except Exception as e:
        logging.error(f"Error reading PDF {pdf_path.name}: {e}")
        return ""

def chunk_text(text: str, file_name: str) -> List[Dict[str, str]]:
    """Chunks text into smaller pieces with metadata."""
    if not text:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True, # Adds character start index for potential future use
    )
    chunks = text_splitter.split_text(text)

    # Create structured chunks with metadata (source filename)
    structured_chunks = [
        {"page_content": chunk, "metadata": {"source": file_name}}
        for chunk in chunks
    ]
    logging.info(f"Split text from {file_name} into {len(structured_chunks)} chunks.")
    return structured_chunks


def process_documents() -> List[Dict[str, str]]:
    """
    Processes all PDF documents in the specified directory:
    1. Finds PDF files.
    2. Extracts text from each PDF.
    3. Chunks the extracted text.
    Returns a list of text chunks (dictionaries with 'page_content' and 'metadata').
    """
    all_chunks = []
    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        logging.warning(f"No PDF files found in {DOCUMENTS_DIR}. Please add some documents.")
        return []

    logging.info(f"Found {len(pdf_files)} PDF files to process.")

    for pdf_path in pdf_files:
        logging.info(f"Processing: {pdf_path.name}")
        raw_text = extract_text_from_pdf(pdf_path)
        if raw_text:
            chunks = chunk_text(raw_text, pdf_path.name)
            all_chunks.extend(chunks)
        else:
            logging.warning(f"Skipping {pdf_path.name} due to extraction errors or empty content.")

    logging.info(f"Total chunks created from all documents: {len(all_chunks)}")
    return all_chunks

if __name__ == '__main__':
    # Example usage: Run this script directly to test processing
    processed_chunks = process_documents()
    if processed_chunks:
        print(f"\nSuccessfully processed {len(processed_chunks)} chunks.")
        # print("\nFirst few chunks:")
        # for i, chunk in enumerate(processed_chunks[:3]):
        #     print(f"--- Chunk {i+1} (Source: {chunk['metadata']['source']}) ---")
        #     print(chunk['page_content'][:200] + "...") # Print start of chunk
    else:
        print("No chunks were processed. Check logs and data/documents folder.")