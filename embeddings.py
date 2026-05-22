from langchain_huggingface import HuggingFaceEmbeddings # Helps to convert the chunks into numbers
from langchain_chroma import Chroma # Helps to CONNECT to the chroma_db where chunks are stored as numbers
from ingest import load_and_chunk
import os
from dotenv import load_dotenv

load_dotenv()

def store_embeddings(pdf_path):
    chunks = load_and_chunk(pdf_path)

    print("⏳ Creating embeddings and storing in ChromaDB...")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print("✅ Embeddings stored in ChromaDB!")
    return vectorstore

if __name__ == "__main__":
    store_embeddings("sample.pdf")