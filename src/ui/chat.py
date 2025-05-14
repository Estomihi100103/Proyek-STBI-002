"""
Chat interface components for Streamlit application
"""
import streamlit as st
from typing import Dict, Any
from ..database.manager import DatabaseManager
from ..retrieval.processor import RetrievalProcessor
from ..retrieval.utils import role_for_streamlit
from ..config import gemini_pro

def render_chat_interface(db: DatabaseManager, 
                         rag_processor: RetrievalProcessor, 
                         retrieval_params: Dict[str, Any]) -> None:
    """Render the chat interface.
    
    Args:
        db: Database manager instance
        rag_processor: Retrieval processor instance
        retrieval_params: Dictionary of retrieval parameters
    """
    st.title("🤖 TalkBot")
    
    # Initialize model and chat history if needed
    model = gemini_pro()
    
    if "chat_history" not in st.session_state or st.session_state['chat_history'] is None:
        st.session_state['chat_history'] = model.start_chat(history=[])
    
    # Extract retrieval parameters
    initial_k = retrieval_params.get("initial_k", 50)
    k = retrieval_params.get("k", 10)
    final_k = retrieval_params.get("final_k", 3)
    
    # Display current conversation for debugging
    conversation_id = st.session_state.get('conversation_id')
    if not conversation_id:
        st.error("No active conversation. Please create a new conversation.")
        return
    
    # Display chat history from database
    display_chat_history(db, conversation_id)
    
    # Get user input
    user_input = st.chat_input("Message TalkBot:")
    if user_input:
        process_user_input(user_input, db, rag_processor, conversation_id, initial_k, k, final_k)

def display_chat_history(db: DatabaseManager, conversation_id: int) -> None:
    """Display the chat history.
    
    Args:
        db: Database manager instance
        conversation_id: ID of the conversation to display
    """
    messages = db.get_conversation_messages(conversation_id)
    for role, content in messages:
        with st.chat_message(role_for_streamlit(role)):
            st.markdown(content)

def process_user_input(user_input: str, 
                      db: DatabaseManager,
                      rag_processor: RetrievalProcessor, 
                      conversation_id: int,
                      initial_k: int,
                      k: int,
                      final_k: int) -> None:
    """Process user input and generate response.
    
    Args:
        user_input: User's message
        db: Database manager instance
        rag_processor: Retrieval processor instance
        conversation_id: ID of the current conversation
        initial_k: Number of initial candidates
        k: Number of BM25 candidates
        final_k: Number of final contexts
    """
    st.chat_message("user").markdown(user_input)
    
    db.add_message(conversation_id, "user", user_input)
    
    with st.spinner("Thinking..."):
        contexts = rag_processor.retrieve_context(
            user_input, 
            k=k, 
            initial_k=initial_k, 
            final_k=final_k
        )
        
        if contexts:
            context_text = "\n".join(contexts)
            augmented_prompt = f"""Context:
{context_text}

Question: {user_input}

Answer:"""
        else:
            augmented_prompt = user_input  
        
        # Generate response
        try:
            response = st.session_state['chat_history'].send_message(augmented_prompt)
            
            # Add response to database
            db.add_message(conversation_id, "assistant", response.text)
            
            # Display response
            with st.chat_message("assistant"):
                st.markdown(response.text)
                
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            # Add error message to database
            db.add_message(conversation_id, "assistant", f"I'm sorry, I encountered an error: {str(e)}")