"""
Main Streamlit application for RAG Chatbot
"""
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import datetime
from dotenv import load_dotenv

from src.config import setup_environment
from src.database.manager import DatabaseManager
from src.retrieval.processor import RetrievalProcessor
from src.ui.chat import render_chat_interface
from src.ui.upload import render_upload_interface

# Load environment variables and setup
setup_environment()

# Set page configuration
st.set_page_config(
    page_title="Chat With Gemi",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

db = DatabaseManager()
rag_processor = RetrievalProcessor(db)

with st.sidebar:
    st.title("Chat History")
    
    user_picked = option_menu(
        None,
        options=["ChatBot", "Document Upload"],
        menu_icon="robot",
        icons=["chat-dots-fill", "file-earmark-arrow-up"],
        default_index=0
    ) ## bagian ini untuk menampilkan sidebar
    
    if st.button("New Conversation"):
        new_conversation_id = db.add_conversation(f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") ## ini berguna untuk menambahakan conversation baru
        st.session_state['conversation_id'] = new_conversation_id ## ini berguna untuk membuat agar state conversation_id menjadi baru
        st.session_state['chat_history'] = None   
        st.rerun()

    conversations = db.get_all_conversations() 
    
    if conversations:
        conversation_titles = [title or f"Chat {cid}" for cid, title in conversations]
        selected_conversation = st.selectbox(
            "Select Conversation",
            options=conversation_titles,
            index=0,
            key="conversation_selector"
        )
        selected_conversation_id = conversations[conversation_titles.index(selected_conversation)][0]
        
        if selected_conversation_id != st.session_state.get('conversation_id'):
            st.session_state['conversation_id'] = selected_conversation_id
            st.session_state['chat_history'] = None  
    else:
        if "conversation_id" not in st.session_state:
            st.session_state['conversation_id'] = db.add_conversation("First Chat")
            st.session_state['chat_history'] = None  

retrieval_params = {
    "initial_k": 50,
    "k": 10,
    "final_k": 3
}

if user_picked == 'ChatBot':
    render_chat_interface(db, rag_processor, retrieval_params)
elif user_picked == 'Document Upload':
    render_upload_interface(rag_processor)