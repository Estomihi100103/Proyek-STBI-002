import sqlite3
import os
from typing import List, Tuple
from ..config import get_db_path

class DatabaseManager:
    """Manages SQLite database operations for the RAG Chatbot."""

    def __init__(self, db_path: str = None):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file. If None, uses config default.
        """
        self.db_path = db_path or get_db_path()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Create conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Create messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)
            # Create document_chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_text TEXT NOT NULL,
                    vector_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def add_conversation(self, title: str) -> int:
        """Add a new conversation to the database.
        
        Args:
            title: Title of the conversation
            
        Returns:
            ID of the new conversation
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO conversations (title) VALUES (?)", (title,))
            conn.commit()
            return cursor.lastrowid

    def get_all_conversations(self) -> List[Tuple[int, str]]:
        """Get all conversations from the database.
        
        Returns:
            List of tuples containing (id, title)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title FROM conversations ORDER BY created_at DESC")
            return cursor.fetchall()

    def add_message(self, conversation_id: int, role: str, content: str) -> None:
        """Add a message to a conversation.
        
        Args:
            conversation_id: ID of the conversation
            role: Role of the message sender ('user' or 'assistant')
            content: Message content
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                (conversation_id, role, content)
            )
            conn.commit()

    def get_messages(self, conversation_id: int) -> List[Tuple[int, int, str, str, str]]:
        """Get all messages for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            List of tuples containing (id, conversation_id, role, content, created_at)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, conversation_id, role, content, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY created_at",
                (conversation_id,)
            )
            return cursor.fetchall()

    def add_document_chunk(self, chunk_text: str, vector_id: str) -> None:
        """Add a document chunk to the database.
        
        Args:
            chunk_text: Text of the document chunk
            vector_id: Vector ID associated with the chunk
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO document_chunks (chunk_text, vector_id) VALUES (?, ?)",
                (chunk_text, vector_id)
            )
            conn.commit()