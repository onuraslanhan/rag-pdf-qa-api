import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Updated imports for optimal compatibility
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="RAG PDF Analysis API")

# Initialize Embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.2,
    max_retries=3
)

vector_store = None


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store
    
    file_path = f"temp_{file.filename}"
    
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        
        vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        return {"message": f"'{file.filename}' processed and indexed successfully!"}
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/ask-question/")
async def ask_question(question: str = Form(...)):
    global vector_store
    
    if not vector_store:
        return {"error": "Please upload a PDF first!"}
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    system_prompt = (
        "You are a helpful assistant. Answer the question in English using ONLY the provided context. "
        "If the information is not in the context, state 'This information is not available in the uploaded document.'\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    answer = rag_chain.invoke(question)
    
    return {
        "question": question,
        "answer": answer
    }