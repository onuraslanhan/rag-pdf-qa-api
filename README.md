# RAG PDF QA API

A FastAPI-based RAG (Retrieval-Augmented Generation) API that lets you upload a PDF and ask questions about its content in English, using LangChain, ChromaDB for vector storage, HuggingFace embeddings, and Google's Gemini model for answer generation.

## Features

- Upload a PDF and index its content into a vector store
- Ask natural language questions about the uploaded document
- Answers are generated using only the document's content (RAG)

## Tech Stack

- FastAPI
- LangChain
- ChromaDB
- HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- Google Gemini (`gemini-3.5-flash`)

## Setup

1. Clone the repository

   git clone https://github.com/onuraslanhan/rag-pdf-qa-api.git
   cd rag-pdf-qa-api

2. Install dependencies

   pip install -r requirements.txt

3. Create a `.env` file in the project root and add your Google API key:

   GOOGLE_API_KEY=your_api_key_here

   You can get a free key at Google AI Studio (https://aistudio.google.com/apikey).

4. Run the server

   uvicorn LLM_API:app --reload

5. Open http://127.0.0.1:8000/docs to test the endpoints.

## Endpoints

- POST /upload-pdf/ — Upload a PDF file to index
- POST /ask-question/ — Ask a question about the uploaded PDF

## License

MIT