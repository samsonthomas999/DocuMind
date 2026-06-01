import streamlit as st
from ingest import load_and_chunk
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="DocuMind", page_icon="🧠", layout="centered")
st.title("🧠 DocuMind")
st.markdown("Ask questions from your PDF documents")
st.divider()

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ File Uploaded Successfully!")

    if st.session_state.processed_file == uploaded_file.name:
        st.info("⚡ Already processed! Using existing data.")
    else:
        with st.spinner("Processing PDF..."):
            chunks = load_and_chunk("temp.pdf")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            st.session_state.processed_file = uploaded_file.name
            st.session_state.vectorstore = vectorstore
        st.success("✅ PDF processed and ready!")

    with st.form(key="question_form", clear_on_submit=True):
        question = st.text_input("Ask a question about your PDF:")
        submitted = st.form_submit_button("Ask")

    if submitted and question:
        with st.spinner("Thinking..."):
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
            relevant_chunks = retriever.invoke(question)
            context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
            llm = ChatGroq(model="llama-3.3-70b-versatile")
            prompt = f"""Use the following context to answer this question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
            response = llm.invoke(prompt)
        sources = list(set([
            f"Page {chunk.metadata.get('page', 0) + 1}"
            for chunk in relevant_chunks
        ]))
        st.session_state.chat_history.append({
            "question": question,
            "answer": response.content,
            "sources": sources
        })

    if st.session_state.chat_history:
        st.divider()
        st.markdown("### 💬 Chat History")
        for chat in st.session_state.chat_history:
            st.markdown(f"**🧑 You:** {chat['question']}")
            st.markdown(f"**🤖 DocuMind:** {chat['answer']}")
            st.caption(f"📄 Sources: {', '.join(chat['sources'])}")
            st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()