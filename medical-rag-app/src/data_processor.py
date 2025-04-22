import os
import logging
from typing import List
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
    
    Returns:
        str: Extracted text from the PDF
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def preprocess_text(text: str) -> List[str]:
    """
    Preprocess text by splitting into sentences.
    
    Args:
        text (str): Input text
    
    Returns:
        List[str]: List of sentences
    """
    # Remove extra whitespace and newlines
    text = " ".join(text.split())
    
    # Split into sentences
    sentences = sent_tokenize(text)
    
    # Filter out very short sentences
    sentences = [sent.strip() for sent in sentences if len(sent.strip()) > 10]
    
    return sentences

def process_pdf_documents(documents_dir: str = "data/documents", 
                           output_dir: str = "data/processed") -> List[str]:
    """
    Process PDF documents in the specified directory.
    
    Args:
        documents_dir (str): Directory containing PDF files
        output_dir (str): Directory to save processed documents
    
    Returns:
        List[str]: List of processed document paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    processed_docs = []
    
    # Iterate through PDF files
    for filename in os.listdir(documents_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(documents_dir, filename)
            
            # Extract text from PDF
            text = extract_text_from_pdf(pdf_path)
            
            # Preprocess text
            sentences = preprocess_text(text)
            
            # Save processed sentences
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_processed.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(sentences))
            
            processed_docs.append(output_path)
            logger.info(f"Processed document: {filename}")
    
    logger.info(f"Total processed documents: {len(processed_docs)}")
    return processed_docs