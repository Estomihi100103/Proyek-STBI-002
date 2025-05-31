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
import requests
from bs4 import BeautifulSoup
import re
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
    contexts = rag_processor.retrieve_context(question, k=3)
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
    
    evaluate_chain = create_evaluate_chain() # panggil fungsi create_evaluate_chain dari src/chains/evaluate.py untuk mendapat object  runnable sequence
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
    
    tavily_client = TavilySearchResults(max_results=10)
    response = tavily_client.invoke({"query": question})
    
    # Ambil URL dari hasil search (cek jika 'url' ada)
    urls_search = [result.get("url") for result in response if result.get("url")]

    # Filter hanya URL yang mengandung domain del.ac.id
    urls_del = [url for url in urls_search if "del.ac.id" in url]
    print(f"Filtered URLs: {urls_del}")
    if not urls_del:
        # Jika tidak ada URL del.ac.id, return None untuk dokumen dan hasil web
        return {
            "question": question,
            "documents": None,
            "conversation_id": conversation_id,
            "web_results": None,
            "solution": None
        }
    
    # lakukan web scraping ke url pertama yang ditemukan
    url_to_escrape = urls_del[0]
    
    # panggil fungsi scrape_webpage untuk mendapatkan konten
    try:
        scraped_content = scrape_webpage(url_to_escrape)
    except ValueError as e:
        print(f"Error: {e}")
        scraped_content = None
    if not scraped_content:
        return {
            "question": question,
            "documents": None,
            "conversation_id": conversation_id,
            "web_results": None,
            "solution": None
        }
        
    # Bagi konten menjadi chunks
    chunks = chunk_text(scraped_content, chunk_size=1024, overlap=100)
    
    # Ubah chunks menjadi vektor
    chunk_vectors = embed_text(chunks)
    
    # Ubah query menjadi vektor
    query_vector = embed_query(question)
    
    
    # cari chunk yang relevan
    relevant_chunks = find_relevant_chunks(query_vector, chunk_vectors, chunks, top_k=2)
    if not relevant_chunks:
        return {
            "question": question,
            "documents": None,
            "conversation_id": conversation_id,
            "web_results": None,
            "solution": None
        }
    
    web_content = "\n".join([chunk for chunk in relevant_chunks])
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
    
    print(f"Relevan Konteks: {documents}")
    
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
        
        
def scrape_webpage(url: str) -> str:
    """Scrape the content of a webpage"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  
        soup= BeautifulSoup(response.text, 'html.parser')
        
        for script in soup(["script", "style", "img", "iframe", "link", 'a', 'h2']): 
            script.decompose() 
            
        # yang berada di dalam class "footer-bg" tidak perlu diambil
        footer_bg = soup.find(class_="footer-bg")
        if footer_bg:
            footer_bg.decompose()
        scraped_text = soup.get_text(separator="\n", strip=True)
        
        scraped_text = "\n".join(line for line in scraped_text.splitlines() if line.strip())
        scraped_text = pre_process_text(scraped_text)
        if not scraped_text:
            raise ValueError("No content found on the page.")
        return scraped_text
    except requests.RequestException as e:
        raise ValueError(f"Failed to retrieve webpage: {str(e)}")
    
    
# fungsi untuk praproses teks
def pre_process_text(text: str) -> str:
    """Pre-process the text by removing extra spaces and newlines."""
    # remove karakter khusus
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"-]', '', text)
    # ubah ke huruf kecil
    text = text.lower()
    # hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    # hapus newline
    text = re.sub(r'\n+', ' ', text).strip()
    # hapus spasi di awal dan akhir
    text = text.strip()
    return text


def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 100) -> List[str]:
    # split teks menjadi chunks, 1 chunk berisi 1024 karakter dengan overlap 100 karakter
    try:
        # Validasi input
        if not text or not isinstance(text, str):
            return []
       
        if chunk_size <= 0:
            return []
        
        # Validasi overlap
        if overlap < 0 or overlap >= chunk_size:
            overlap = 0  # Set overlap ke 0 jika tidak valid
       
        # Buat list untuk menyimpan chunks
        chunks = []
        
        # Hitung step size (jarak antar chunk)
        step_size = chunk_size - overlap
        
        # Loop untuk memotong teks dengan overlap
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Jika chunk ini sudah mencakup seluruh sisa teks, break
            if end >= len(text):
                break
                
            # Pindah ke posisi berikutnya dengan mempertimbangkan overlap
            start += step_size
        return chunks
       
    except Exception as e:
        print(f"Error chunking text: {e}")
        return []

# fungsi untuk mengubah teks hasil crawl menjadi vektor
def embed_text(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[np.ndarray]:
    """Embed texts into vector representations using Google Generative AI.
    
    Args:
        texts: List of texts to embed
        task_type: Type of embedding task (e.g., RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY)
    
    Returns:
        List of vector embeddings
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            task_type=task_type
        )
        vectors = embeddings.embed_documents(texts)
        return vectors
    except Exception as e:
        print(f"Error embedding texts: {e}")
        return []


# fungsi untuk mengubah pertanyaan menjadi vektor
def embed_query(query: str, task_type: str = "RETRIEVAL_QUERY") -> np.ndarray:
    """Embed a single query into a vector representation.
    
    Args:
        query: Query text to embed
        task_type: Type of embedding task
    
    Returns:
        Vector embedding of the query
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            task_type=task_type
        )
        vector = embeddings.embed_query(query)
        return vector
    except Exception as e:
        print(f"Error embedding query: {e}")
        return None

# fungsi untuk melakukan pencarian antara vektor pertanyaan dan vektor teks hasil crawl
def find_relevant_chunks(query_vector: np.ndarray, 
                                  chunk_vectors: List[np.ndarray], 
                                  chunks: List[str], 
                                  top_k: int = 3) -> List[str]:
    """Find the most relevant text chunks based on cosine similarity.
    
    Args:
        query_vector: Vector representation of the query
        chunk_vectors: List of vector representations of text chunks
        chunks: List of original text chunks
        top_k: Number of top relevant chunks to return
    
    Returns:
        List of relevant text chunks
    """
    try:
        if not query_vector or not chunk_vectors or not chunks:
            return []
        
        # Hitung cosine similarity
        similarities = cosine_similarity([query_vector], chunk_vectors)[0]
        
        # Urutkan berdasarkan similarity
        sorted_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Ambil chunk yang relevan
        relevant_chunks = [chunks[i] for i in sorted_indices]  
        return relevant_chunks
    except Exception as e:
        print(f"Error finding relevant chunks: {e}")
        return []