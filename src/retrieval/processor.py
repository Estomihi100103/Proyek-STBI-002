"""
Retrieval processor for RAG Chatbot application
"""
import os
import uuid
import re
from typing import List, Dict, Any, Optional, Tuple
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from ..config import get_api_key, get_chroma_path, get_embedding_model, get_chunking_config
from ..database.manager import DatabaseManager
from .utils import extract_text_from_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/retrieval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RetrievalProcessor:
    """Handles retriever pipeline: document processing, embedding, and retrieval."""

    def __init__(self, db: DatabaseManager, persist_directory: str = None):
        """Initialize retrieval processor with database and vector store.
        
        Args:
            db: Database manager instance
            persist_directory: Directory for Chroma DB persistence. If None, uses config default.
        """
        self.db = db
        self.persist_directory = persist_directory or get_chroma_path()
        
        # Ensure persistence directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=get_embedding_model(),
            google_api_key=get_api_key()
        )
        
        # Initialize Chroma vector store
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="rag_collection"
        )
        
        # Log number of documents in vector store
        collection_count = self.vector_store._collection.count()
        logger.info(f"Vector store initialized with {collection_count} documents")
        
        # Initialize text splitter with config
        chunking_config = get_chunking_config()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunking_config["chunk_size"],
            chunk_overlap=chunking_config["chunk_overlap"],
            length_function=len
        )

    def process_document(self, file: bytes, file_type: str) -> List[str]:
        """Process document: extract text, chunk, embed, and store.
        
        Args:
            file: File bytes
            file_type: Type of file (pdf, docx, txt)
            
        Returns:
            List of chunk IDs
        """
        try:
            # Extract text
            text = extract_text_from_file(file, file_type)
            logger.info(f"Extracted text from {file_type} file, length: {len(text)}")
            
            # Chunk text
            chunks = self.text_splitter.split_text(text)
            
            print(f"Split text into {len(chunks)} chunks")
            
            logger.info(f"Split text into {len(chunks)} chunks")
            
            # Generate embeddings and store in Chroma DB
            vector_ids = [str(uuid.uuid4()) for _ in chunks]
            self.vector_store.add_texts(
                texts=chunks,
                ids=vector_ids,
                metadatas=[{"id": vid} for vid in vector_ids]
            )
            logger.info(f"Added {len(chunks)} chunks to vector store")
            
            # Store chunks in SQLite
            for chunk, vector_id in zip(chunks, vector_ids):
                self.db.add_document_chunk(chunk, vector_id)
            
            return vector_ids
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", exc_info=True)
            raise ValueError(f"Error processing document: {str(e)}")

    def retrieve_context(self, 
                        query: str, 
                        k: int = 3) -> List[str]:
        """Retrieve relevant contexts using BM25 retrieval approach.
        
        Args:
            query: User query
            k: Number of final contexts to return (default=2)
            
        Returns:
            List of relevant context passages
        """
        try:            
            # Step 1: Initial dense retrieval
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=5  # Keep initial retrieval broad
            )                        
            # Check if no results were found
            if not results:
                logger.warning("No documents found in initial retrieval")
                return []
            
            # Store text, score, and vector ID
            dense_contexts = [(doc.page_content, score, doc.metadata.get('id', None)) 
                            for doc, score in results]
            
            # Step 2: BM25 indexing
            tokenized_chunks = [word_tokenize(content.lower()) for content, _, _ in dense_contexts]
            bm25 = BM25Okapi(tokenized_chunks)
            
            # Step 3: BM25 ranking
            tokenized_query = word_tokenize(query.lower())
            bm25_scores = bm25.get_scores(tokenized_query)
            
            # Step 4: Sort by BM25 score and select top k
            sorted_results = sorted(zip(dense_contexts, bm25_scores), 
                                key=lambda x: x[1], reverse=True)[:k]
            final_contexts = [content for (content, _, _), _ in sorted_results]
            logger.info(f"BM25 ranking selected {len(final_contexts)} contexts")
            
            return final_contexts
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}", exc_info=True)
            return []