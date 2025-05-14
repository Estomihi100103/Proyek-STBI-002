"""
Utility functions for retrieval operations
"""
import re
from io import BytesIO
import PyPDF2
from docx import Document

def extract_text_from_file(file: bytes, file_type: str) -> str:
    """Extract text from uploaded file based on its type.
    
    Args:
        file: File bytes
        file_type: Type of file (pdf, docx, txt)
        
    Returns:
        Extracted text content
    
    Raises:
        ValueError: If file type is unsupported or extraction fails
    """
    try:
        if file_type == "pdf":
            return extract_text_from_pdf(file)
        elif file_type == "docx":
            return extract_text_from_docx(file)
        elif file_type == "txt":
            return file.read().decode("utf-8")
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        raise ValueError(f"Error extracting text: {str(e)}")

def extract_text_from_pdf(file: bytes) -> str:
    """Extract text from PDF file.
    
    Args:
        file: PDF file bytes
        
    Returns:
        Extracted text content
    """
    reader = PyPDF2.PdfReader(BytesIO(file.read()))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    
    # Preprocess text
    text = preprocess_text(text)
    return text

def extract_text_from_docx(file: bytes) -> str:
    """Extract text from DOCX file.
    
    Args:
        file: DOCX file bytes
        
    Returns:
        Extracted text content
    """
    doc = Document(BytesIO(file.read()))
    text = "\n".join([para.text for para in doc.paragraphs])
    
    # Preprocess text
    text = preprocess_text(text)
    return text

def preprocess_text(text: str) -> str:
    """Preprocess extracted text.
    
    Args:
        text: Raw text to preprocess
        
    Returns:
        Preprocessed text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove any non-printable characters
    text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\t'])
    
    # remove karakter khusus
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"-]', '', text)
    
    # buat huruf kecil
    text = text.lower()
    
    return text

def role_for_streamlit(user_role: str) -> str:
    """Convert model role to Streamlit-compatible role.
    
    Args:
        user_role: Original role (e.g., 'model', 'user')
        
    Returns:
        Streamlit-compatible role ('assistant' or 'user')
    """
    return 'assistant' if user_role == 'model' else user_role