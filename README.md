# RAG Document & Web QA API

A FastAPI-based RAG (Retrieval-Augmented Generation) API that lets you upload PDFs, Word documents (DOCX), or Web URLs and ask questions about their content. It uses LangChain, ChromaDB for persistent vector storage, HuggingFace embeddings, and Google's Gemini model for accurate, source-backed answer generation.

## 🚀 Features

- **Multi-Format Support:** Upload PDF files, DOCX files, or provide Web URLs.
- **Smart Web Scraping:** Uses Trafilatura and BeautifulSoup to extract clean text from complex websites (like Wikipedia).
- **Persistent Vector Store:** Documents are saved to a local `chroma_db` folder, meaning you don't lose your indexed files when the server restarts.
- **Source Tracking:** Answers include the exact source (file name or URL) the information was retrieved from.
- **Strict RAG (No Hallucination):** Answers are generated *strictly* using the uploaded context. If the information is missing, the AI explicitly states it.
- **Multilingual/Turkish Support:** Capable of understanding questions and generating accurate responses in Turkish.

## 🛠️ Tech Stack

- **Framework:** FastAPI
- **LLM Orchestration:** LangChain
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **LLM:** Google Gemini (`gemini-3.5-flash`)
- **Web Scraping:** Trafilatura & BeautifulSoup4

## ⚙️ Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
