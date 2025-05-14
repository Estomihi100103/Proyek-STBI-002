# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit and Google's Gemini API.

## Features

- Chat interface powered by Google's Gemini AI
- Document upload and processing (PDF, DOCX, TXT)
- Hybrid retrieval system (dense + sparse)
- Conversation history management
- Configurable retrieval parameters

## Project Structure

```
information-retrieval/
├── app.py                # Main Streamlit application entry point
├── .env                  # Environment variables
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
├── src/                  # Source code directory
│   ├── __init__.py       # Makes src a Python package
│   ├── database/         # Database-related code
│   │   ├── __init__.py
│   │   └── manager.py    # DatabaseManager class
│   ├── retrieval/        # Retrieval-related code
│   │   ├── __init__.py
│   │   ├── processor.py  # RetrievalProcessor class
│   │   └── utils.py      # Utility functions for retrieval
│   ├── ui/               # UI components
│   │   ├── __init__.py
│   │   ├── chat.py       # Chat interface components
│   │   └── upload.py     # Document upload components
│   └── config.py         # Configuration settings
├── data/                 # Data directory
│   ├── chroma_db/        # Chroma database files
│   └── sqlite_db/        # SQLite database files
└── logs/                 # Log files
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/information-retrieval.git
   cd information-retrieval
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv stbiproject
   source stbiproject/bin/activate  # On Windows: stbiproject\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your Google API key:
   ```
   api_key=your_google_api_key_here
   ```

## Usage

1. Run the application:
   ```
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Use the navigation menu to switch between the chat interface and document upload

## Retrieval Process

The system uses a hybrid retrieval approach:

1. Initial dense retrieval with Chroma DB to get candidates
2. BM25 re-ranking to filter candidates
3. Final dense retrieval for the best context passages

This hybrid approach helps balance semantic understanding with keyword precision.

## Configuration

You can customize the application by modifying the values in `src/config.py` or by setting environment variables.

Key parameters:
- `chunk_size`: Size of document chunks (default: 1024)
- `chunk_overlap`: Overlap between chunks (default: 100)
- `embedding_model`: Model used for embeddings (default: models/text-embedding-004)
- `llm_model`: LLM model for generation (default: gemini-2.0-flash)