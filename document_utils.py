import tempfile
from pdf2image import convert_from_bytes
from pypdf import PdfReader
from document_ai_agents.logger import logger
import asyncio
from typing import List
from pathlib import Path

def extract_images_from_pdf(pdf_path: str) -> List[Image.Image]:
    """
    Extract images from a PDF file asynchronously.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        List[Image.Image]: List of PIL Image objects extracted from the PDF.
    
    Raises:
        FileNotFoundError: If the PDF file does not exist.
        Exception: If image extraction fails.
    """
    logger.info(f"Extracting images from PDF: {pdf_path}")
    if not Path(pdf_path).is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        with open(pdf_path, "rb") as f:
            with tempfile.TemporaryDirectory() as path:
                logger.info(f"Converting PDF to images using temporary directory: {path}")
                images = convert_from_bytes(f.read(), output_folder=path, fmt="jpeg")
                logger.info(f"Extracted {len(images)} images from the PDF.")
                return images
    except Exception as e:
        logger.error(f"Failed to extract images from PDF: {e}")
        raise

def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """
    Extract text from a PDF file asynchronously.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        List[str]: List of text strings extracted from each page of the PDF.
    
    Raises:
        FileNotFoundError: If the PDF file does not exist.
        Exception: If text extraction fails.
    """
    logger.info(f"Extracting text from PDF: {pdf_path}")
    if not Path(pdf_path).is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            logger.info(f"Extracting text from {len(reader.pages)} pages.")
            texts = [page.extract_text() or "" for page in reader.pages]  # Handle empty text
            logger.info(f"Extracted text from {len(texts)} pages.")
            return [text.strip() for text in texts if text.strip()]  # Remove empty or whitespace-only strings
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise

async def async_extract_images_from_pdf(pdf_path: str) -> List[Image.Image]:
    """
    Asynchronously extract images from a PDF file using a thread pool.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        List[Image.Image]: List of PIL Image objects extracted from the PDF.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, extract_images_from_pdf, pdf_path)

async def async_extract_text_from_pdf(pdf_path: str) -> List[str]:
    """
    Asynchronously extract text from a PDF file using a thread pool.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        List[str]: List of text strings extracted from each page of the PDF.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, extract_text_from_pdf, pdf_path)
