"""
Document upload interface components for Streamlit application
"""
import streamlit as st
from typing import List
import os
from ..retrieval.processor import RetrievalProcessor

def render_upload_interface(rag_processor: RetrievalProcessor) -> None:
    """Render the document upload interface.
    
    Args:
        rag_processor: Retrieval processor instance
    """
    st.title("📄 Document Upload")
    st.write("Upload PDFs, Word documents, or text files to process for Retrieval Augmented Generation (RAG).")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose documents to upload",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        process_uploaded_files(uploaded_files, rag_processor)
    
    # Display document stats
    with st.expander("Document Statistics", expanded=False):
        display_document_stats(rag_processor)

def process_uploaded_files(uploaded_files: List, rag_processor: RetrievalProcessor) -> None:
    """Process uploaded files.
    
    Args:
        uploaded_files: List of uploaded files
        rag_processor: Retrieval processor instance
    """
    for uploaded_file in uploaded_files:
        try:
            # Determine file type
            file_type = uploaded_file.name.split(".")[-1].lower()
            if file_type not in ["pdf", "docx", "txt"]:
                st.error(f"Unsupported file type for {uploaded_file.name}. Please upload PDFs, Word documents, or text files.")
                continue
            
            # Process document with progress indicator
            with st.status(f"Processing {uploaded_file.name}...", expanded=True) as status:
                st.write(f"Extracting text from {file_type.upper()} file...")
                vector_ids = rag_processor.process_document(uploaded_file, file_type)
                
                st.write(f"Added {len(vector_ids)} chunks to the knowledge base")
                status.update(label=f"Processed {uploaded_file.name}", state="complete")
            
            st.success(f"Successfully processed {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

def display_document_stats(rag_processor: RetrievalProcessor) -> None:
    """Display statistics about indexed documents.
    
    Args:
        rag_processor: Retrieval processor instance
    """
    try:
        # Get collection statistics
        collection_stats = rag_processor.vector_store._collection.count()
        st.metric("Total Document Chunks", collection_stats)
        
        # Display additional information about the vector store
        st.write(f"Vector store location: {os.path.abspath(rag_processor.persist_directory)}")
        st.write(f"Embedding model: {rag_processor.embeddings.model}")
        
    except Exception as e:
        st.warning(f"Could not retrieve document statistics: {str(e)}")