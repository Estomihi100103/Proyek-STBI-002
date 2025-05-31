"""
Configuration settings for the RAG Chatbot application
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv
import nltk

# Default configuration values
DEFAULT_CONFIG = {
    "db_path": "data/sqlite_db/chat_app.db",
    "chroma_path": "data/chroma_db",
    "embedding_model": "models/text-embedding-004",
    "llm_model": "gemini-2.0-flash",
    "chunk_size": 1024,
    "chunk_overlap": 100
}

def setup_environment():
    """Initialize environment settings and dependencies"""
    # Create necessary directories
    os.makedirs("data/sqlite_db", exist_ok=True)
    os.makedirs("data/chroma_db", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Load environment variables
    load_dotenv()
    
    # Configure Gemini API
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set 'GOOGLE_API_KEY' in .env file.")
    genai.configure(api_key=api_key)
    
    # Download NLTK data if needed
    nltk.download('punkt', quiet=True)

def get_db_path():
    """Get the database file path"""
    return os.getenv("DB_PATH", DEFAULT_CONFIG["db_path"])

def get_chroma_path():
    """Get the Chroma DB directory path"""
    return os.getenv("CHROMA_PATH", DEFAULT_CONFIG["chroma_path"])

def get_api_key():
    """Get the Google API key"""
    return os.getenv("GOOGLE_API_KEY")

def get_embedding_model():
    """Get the embedding model name"""
    return os.getenv("EMBEDDING_MODEL", DEFAULT_CONFIG["embedding_model"])

def get_llm_model():
    """Get the LLM model name"""
    return os.getenv("LLM_MODEL", DEFAULT_CONFIG["llm_model"])

def get_chunking_config():
    """Get text chunking configuration"""
    return {
        "chunk_size": int(os.getenv("CHUNK_SIZE", DEFAULT_CONFIG["chunk_size"])),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", DEFAULT_CONFIG["chunk_overlap"]))
    }

def gemini_pro():
    """Initialize and return the Gemini Pro model"""
    return genai.GenerativeModel(get_llm_model())