# MedChatBot

MedChatBot is a retrieval-augmented medical chatbot that uses a Llama 3.2 model to generate context-aware medical advice. It retrieves similar QA pairs from a MedQuad-based FAISS index and uses them to guide answer generation.

## Features
- **RAG:** Retrieves background context from similar medical Q&A pairs.
- **Backend:** FastAPI with Transformers, PyTorch, FAISS, SentenceTransformers, and SQLAlchemy.
- **Frontend:** Next.js (TypeScript) with Material-UI for a modern, responsive UI.
- **Logging:** Saves user queries and responses to a database.

## Technologies
**Backend:** FastAPI, Uvicorn, Transformers, PyTorch, FAISS, SentenceTransformers, SQLAlchemy  
**Frontend:** Next.js, TypeScript, Material-UI

## Setup

### Backend
1. **Clone the repository** and create a Python virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   ```