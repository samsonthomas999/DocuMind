from langchain_groq import ChatGroq # Connects to Groq AI for answering
from langchain_huggingface import HuggingFaceEmbeddings # Helps to convert the chunks into numbers
from langchain_chroma import Chroma # Helps to connect to the chroma_db where chunks are stored as numbers
from langchain_core.prompts import ChatPromptTemplate # Helps to create a prompt template
import os
from dotenv import load_dotenv

load_dotenv()

def ask_question(question):
    # Load embeddings — same model used in embeddings.py
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Connect to existing chroma_db
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    # Search chroma_db for relevant chunks
    # k=3 means fetch top 3 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_chunks = retriever.invoke(question)

    # Combine the 3 chunks into one context paragraph
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    # Load Groq AI
    llm = ChatGroq(model="llama-3.3-70b-versatile")

    # Create prompt — send chunks + question to Gemini
    prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""

    # Send to Gemini AI and get answer back
    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    print("🧠 DocuMind is ready!")
    question = input("You: ")
    answer = ask_question(question)
    print(f"\n🤖 DocuMind: {answer}")


# Ingest file -> 📄 PDF → chunks
# Embeddings file -> 📄 PDF → chunks → numbers → stored in chroma_db
# Rag file :
# ❓ User asks a question
#      ↓
# 🔍 Search chroma_db for relevant chunks
#      ↓
# 🤖 Send those chunks to Gemini AI
#      ↓
# 💬 Get a smart answer back