import streamlit as st
from typing import Dict, Any, List
from src.database import DatabaseManager
from src.retrieval import RetrievalProcessor
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from src.state import GraphState
from src.chains.evaluate import create_evaluate_chain
from src.chains.generate_answer import create_generate_chain
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from src.config import get_api_key

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
    
    # Initialize model
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=get_api_key(),
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    # Initialize chat history if needed
    if "chat_history" not in st.session_state or st.session_state['chat_history'] is None:
        st.session_state['chat_history'] = []
    
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
        process_user_input_langgraph(user_input, db, rag_processor, conversation_id, initial_k, k, final_k)

def display_chat_history(db: DatabaseManager, conversation_id: int) -> None:
    """Display the chat history for the given conversation ID.
    
    Args:
        db: Database manager instance
        conversation_id: ID of the conversation
    """
    messages = db.get_messages(conversation_id)
    for message in messages:
        role = message[2]
        content = message[3]
        with st.chat_message(role):
            st.markdown(content)

def create_workflow(rag_processor: RetrievalProcessor, initial_k: int, k: int, final_k: int) -> StateGraph:
    """Create the LangGraph workflow for processing user input."""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve", lambda state: retrieve(state, rag_processor, initial_k, k, final_k))
    workflow.add_node("evaluate", evaluate)
    workflow.add_node("search_online", search_online)
    workflow.add_node("generate", generate)
    
    # Set entry point and edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        lambda state: "search_online" if not state["documents"] else "generate",
        {
            "search_online": "search_online",
            "generate": "generate"
        }
    )
    workflow.add_edge("search_online", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

def retrieve(state: GraphState, rag_processor: RetrievalProcessor, initial_k: int, k: int, final_k: int) -> GraphState:
    """Retrieve documents from the vector database.
    
    Args:
        state: Current graph state
        rag_processor: Retrieval processor instance
        initial_k, k, final_k: Retrieval parameters
    
    Returns:
        Updated graph state
    """
    question = state["question"]
    conversation_id = state["conversation_id"]
    contexts = rag_processor.retrieve_context(question, k=k, initial_k=initial_k, final_k=final_k)
    return {
        "question": question,
        "documents": contexts,
        "conversation_id": conversation_id,
        "web_results": None,
        "solution": None
    }

def evaluate(state: GraphState) -> GraphState:
    """Evaluate document relevance.
    
    Args:
        state: Current graph state
    
    Returns:
        Updated graph state
    """
    question = state["question"]
    documents = state["documents"]
    conversation_id = state["conversation_id"]
    
    evaluate_chain = create_evaluate_chain()
    filtered_docs = []
    
    for doc in documents:
        response = evaluate_chain.invoke({"question": question, "document": doc})
        if response.score.lower() == "yes":
            filtered_docs.append(doc)
    
    return {
        "question": question,
        "documents": filtered_docs,
        "conversation_id": conversation_id,
        "web_results": None,
        "solution": None
    }

def search_online(state: GraphState) -> GraphState:
    """Search online using Tavily if no relevant documents are found.
    
    Args:
        state: Current graph state
    
    Returns:
        Updated graph state
    """
    
    question = state["question"]
    conversation_id = state["conversation_id"]
    
    tavily_client = TavilySearchResults(max_results=2)
    response = tavily_client.invoke({"query": question})
    web_content = "\n".join([element["content"] for element in response])
    web_doc = [web_content]
    
    return {
        "question": question,
        "documents": web_doc,
        "conversation_id": conversation_id,
        "web_results": web_content,
        "solution": None
    }

def generate(state: GraphState) -> GraphState:
    """Generate an answer based on documents or web results.
    
    Args:
        state: Current graph state
    
    Returns:
        Updated graph state
    """
    question = state["question"]
    documents = state["documents"]
    conversation_id = state["conversation_id"]
    
    context = "\n".join(documents) if documents else "No relevant information found."
    generate_chain = create_generate_chain()
    solution = generate_chain.invoke({"context": context, "question": question})
    
    return {
        "question": question,
        "documents": documents,
        "conversation_id": conversation_id,
        "web_results": state["web_results"],
        "solution": solution
    }

def process_user_input_langgraph(user_input: str, 
                                db: DatabaseManager,
                                rag_processor: RetrievalProcessor, 
                                conversation_id: int,
                                initial_k: int,
                                k: int,
                                final_k: int) -> None:
    """Process user input using LangGraph workflow.
    
    Args:
        user_input: User's question
        db: Database manager instance
        rag_processor: Retrieval processor instance
        conversation_id: ID of the current conversation
        initial_k, k, final_k: Retrieval parameters
    """
    try:
        # Display user message
        st.chat_message("user").markdown(user_input)
        db.add_message(conversation_id, "user", user_input)
        
        # Create and invoke graph
        graph = create_workflow(rag_processor, initial_k, k, final_k)
        result = graph.invoke({
            "question": user_input,
            "conversation_id": conversation_id,
            "documents": [],
            "web_results": None,
            "solution": None
        })
        
        # Display and save assistant response
        solution = result["solution"]
        with st.chat_message("assistant"):
            st.markdown(solution)
        db.add_message(conversation_id, "assistant", solution)
        
    except Exception as e:
        st.error(f"Error processing input: {str(e)}")
        db.add_message(conversation_id, "assistant", f"I'm sorry, I encountered an error: {str(e)}")