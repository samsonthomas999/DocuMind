import streamlit as st
import os
import shutil
from ingest import load_and_chunk
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(page_title="DocuMind", page_icon="🧠", layout="centered")

# Title
st.title("🧠 DocuMind")
st.markdown("Ask questions from your PDF documents")
st.divider()

# Sticky notes — remember between reruns
if "processed_file" not in st.session_state:  # For file
    st.session_state.processed_file = None

if "vectorstore" not in st.session_state:     # For contents inside the file
    st.session_state.vectorstore = None

# NEW — blank paper for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# PDF Upload
uploaded_file = st.file_uploader("Upload your PDF", type="pdf") # creates the upload box

if uploaded_file:
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ File Uploaded Successfully!")

    # Check if already processed
    if st.session_state.processed_file == uploaded_file.name:
        st.info("⚡ Already processed! Using existing data.")

    else:
        # Process PDF for the first time
        with st.spinner("Processing PDF..."):
            chunks = load_and_chunk("temp.pdf")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="./chroma_db",
                collection_name=uploaded_file.name.replace(".pdf", "").replace(" ", "_").lower()
            )
            # Save to sticky notes
            st.session_state.processed_file = uploaded_file.name
            st.session_state.vectorstore = vectorstore

        st.success("✅ PDF processed and ready!")

    # MOD — using form so text input clears properly after submit
    with st.form(key="question_form", clear_on_submit=True):
        question = st.text_input("Ask a question about your PDF:")
        submitted = st.form_submit_button("Ask")

    # Only run when form is submitted with a question
    if submitted and question:
        with st.spinner("Thinking..."):
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
            relevant_chunks = retriever.invoke(question)
            context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

            llm = ChatGroq(model="llama-3.3-70b-versatile")
            prompt = f"""Use the following context to answer this question.

Context:
{context}

Question: {question}

Answer:"""
            response = llm.invoke(prompt)

        # NEW — add to chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": response.content
        })

    # NEW — display all chat history
    if st.session_state.chat_history:
        st.divider()
        st.markdown("### 💬 Chat History")
        for chat in st.session_state.chat_history:
            st.markdown(f"**🧑 You:** {chat['question']}")
            st.markdown(f"**🤖 DocuMind:** {chat['answer']}")
            st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()