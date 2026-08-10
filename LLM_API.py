import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="RAG PDF Analiz API")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.2,
    max_retries=3
)

vector_store = None

@app.post("/pdf-yukle/")
async def pdf_yukle(file: UploadFile = File(...)):
    global vector_store
    
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    os.remove(file_path)
    return {"mesaj": f"'{file.filename}' başarıyla okundu ve hafızaya alındı!"}


@app.post("/soru-sor/")
async def soru_sor(soru: str = Form(...)):
    global vector_store
    
    if not vector_store:
        return {"hata": "Önce bir PDF yüklemelisiniz!"}
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    system_prompt = (
        "Sen yardımcı bir asistansın. Sadece sana verilen bağlamı (context) kullanarak soruya Türkçe cevap ver. "
        "Eğer bilgi verilen metinde yoksa 'Bu bilgi yüklenen dokümanda bulunmuyor' de.\n\n"
        "Metin:\n{context}\n\nSoru: {question}"
    )
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    cevap = rag_chain.invoke(soru)
    
    return {
        "soru": soru,
        "cevap": cevap
    }