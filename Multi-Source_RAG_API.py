import os
from pathlib import Path
import statistics
from langchain_core.documents import Document
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

import trafilatura
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="RAG PDF Analysis API")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.2,
    max_retries=3
)

vector_store = None

def get_or_create_vector_store():
    global vector_store
    if vector_store is None:
        vector_store = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
    return vector_store


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    
    file_path = f"temp_{file.filename}"
    
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        
        store = get_or_create_vector_store()
        store.add_documents(splits)
        
        return {"message": f"'{file.filename}' processed and indexed successfully!"}
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/upload-docx/")
async def upload_docx(file: UploadFile = File(...)):

    file_path = f"temp_{file.filename}"
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        
        store = get_or_create_vector_store()
        store.add_documents(splits)
        
        return {"message": f"'{file.filename}' processed and indexed successfully!"}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def load_url_clean(url: str) -> list[Document]:
    text = None
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        text = trafilatura.extract(downloaded)
    if not text or len(text.strip()) == 0:
        loader = WebBaseLoader(url)
        docs = loader.load()
        if docs and len(docs[0].page_content.strip()) > 0:
            text = docs[0].page_content
        else:
            raise ValueError(f"Could not get content from URL: {url}")
    
    return [Document(page_content=text, metadata={"source": url})]

@app.post("/upload-url/")
async def upload_url(url: str = Form(...)):
    
    docs = load_url_clean(url)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    
    store = get_or_create_vector_store()
    store.add_documents(splits)
    
    return {"message": f"'{url}' processed and indexed successfully!"}

@app.post("/ask-question/")
async def ask_question(question: str = Form(...)):
    store = get_or_create_vector_store()
    
    retrieved_docs_with_scores = store.similarity_search_with_relevance_scores(question, k=5)

    for doc, score in retrieved_docs_with_scores:
        print(f"Score: {score:.3f} | Source: {doc.metadata.get('source')} | Text: {doc.page_content[:80]}")

    scores = [score for doc, score in retrieved_docs_with_scores]
    max_score = max(scores)

    relevant_docs = [
        doc for doc, score in retrieved_docs_with_scores 
        if score >= max_score * 0.7 and score > 0.02
    ]
    
    if not relevant_docs:
        return {
            "question": question,
            "answer": "That information is not available in the uploaded document.",
            "sources": []
        }
    
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    sources = list(set(doc.metadata.get("source", "unknown") for doc in relevant_docs))
    
    system_prompt = (
        "You are a helpful assistant. Answer the question in English using ONLY the provided context. "
        "If the information is not in the context, state 'This information is not available in the uploaded document.'\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }
