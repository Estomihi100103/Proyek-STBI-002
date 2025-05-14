"""
Database manager for the RAG Chatbot application
"""
import sqlite3
from datetime import datetime
from typing import List, Tuple
import os
from ..config import get_db_path

class DatabaseManager:
    """Manages SQLite database operations for the chat application."""

    def __init__(self, db_name: str = None):
        """Initialize database connection and create tables.
        
        Args:
            db_name: Optional database file path. If None, uses the default path from config.
        """
        self.db_name = db_name or get_db_path()
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_name), exist_ok=True)
        
        self._init_db()

    def _init_db(self) -> None:
        """Create necessary tables if they don't exist."""
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            
            # Create Conversation table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Conversation (
                    conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create Message table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Message (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES Conversation(conversation_id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS DocumentChunk (
                    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    vector_id TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()

    def add_conversation(self, title: str = None) -> int:
        """Add a new conversation and return its conversation_id.
        
        Args:
            title: Optional title for the conversation.
            
        Returns:
            conversation_id: The ID of the newly created conversation
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO Conversation (title) VALUES (?)",
                (title,)
            )
            conn.commit()
            return cursor.lastrowid

    def add_message(self, conversation_id: int, role: str, content: str) -> int:
        """Add a new message to a conversation.
        
        Args:
            conversation_id: The ID of the conversation to add the message to
            role: The role of the message sender (user or assistant)
            content: The message content
            
        Returns:
            message_id: The ID of the newly created message
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO Message (conversation_id, role, content) VALUES (?, ?, ?)",
                (conversation_id, role, content)
            )
            conn.commit()
            return cursor.lastrowid

    def add_document_chunk(self, content: str, vector_id: str) -> int:
        """Add a document chunk to the database.
        
        Args:
            content: The text content of the chunk
            vector_id: The ID of the vector in the vector store
            
        Returns:
            chunk_id: The ID of the newly created chunk
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO DocumentChunk (content, vector_id) VALUES (?, ?)",
                (content, vector_id)
            )
            conn.commit()
            return cursor.lastrowid

    def get_conversation_messages(self, conversation_id: int) -> List[Tuple[str, str]]:
        """Retrieve all messages for a given conversation.
        
        Args:
            conversation_id: The ID of the conversation to get messages for
            
        Returns:
            List of (role, content) tuples representing messages in the conversation
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content FROM Message WHERE conversation_id = ? ORDER BY timestamp",
                (conversation_id,)
            )
            return cursor.fetchall()

    def get_all_conversations(self) -> List[Tuple[int, str]]:
        """Retrieve all conversations.
        
        Returns:
            List of (conversation_id, title) tuples for all conversations
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT conversation_id, title FROM Conversation ORDER BY created_at DESC"
            )
            return cursor.fetchall()
    
    def update_conversation_title(self, conversation_id: int, title: str) -> None:
        """Update the title of a conversation.
        
        Args:
            conversation_id: The ID of the conversation to update
            title: The new title for the conversation
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE Conversation SET title = ? WHERE conversation_id = ?",
                (title, conversation_id)
            )
            conn.commit()
    
    def delete_conversation(self, conversation_id: int) -> None:
        """Delete a conversation and all its messages.
        
        Args:
            conversation_id: The ID of the conversation to delete
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            # Delete messages first (foreign key constraint)
            cursor.execute(
                "DELETE FROM Message WHERE conversation_id = ?",
                (conversation_id,)
            )
            # Then delete the conversation
            cursor.execute(
                "DELETE FROM Conversation WHERE conversation_id = ?",
                (conversation_id,)
            )
            conn.commit()