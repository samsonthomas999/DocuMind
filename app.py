import streamlit as st
from ingest import load_and_chunk
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="DocuMind", page_icon="🧠", layout="centered")

# ── Sidebar ──────────────────────────────────────
st.sidebar.title("📁 DocuMind")
st.sidebar.markdown("Upload your PDFs below")
st.sidebar.divider()

uploaded_files = st.sidebar.file_uploader(
    "Upload your PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    st.sidebar.divider()
    st.sidebar.markdown("**📄 Loaded Files:**")
    for f in uploaded_files:
        st.sidebar.caption(f"✅ {f.name}")

st.sidebar.divider()
st.sidebar.markdown("Built with LangChain, FAISS & Groq")

# ── Main Area ─────────────────────────────────────
st.title("🧠 DocuMind")
st.markdown("Ask questions from your PDF documents")
st.divider()

if "processed_files" not in st.session_state:
    st.session_state.processed_files = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:
    file_names = sorted([f.name for f in uploaded_files])

    if st.session_state.processed_files == file_names:
        st.info("⚡ Already processed! Using existing data.")
    else:
        with st.spinner("Processing PDFs..."):
            all_chunks = []
            for uploaded_file in uploaded_files:
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.read())
                chunks = load_and_chunk(uploaded_file.name)
                all_chunks = all_chunks + chunks

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(all_chunks, embeddings)
            st.session_state.processed_files = file_names
            st.session_state.vectorstore = vectorstore
        st.success(f"✅ {len(uploaded_files)} PDF(s) processed and ready!")

    with st.form(key="question_form", clear_on_submit=True):
        question = st.text_input("Ask a question about your PDFs:")
        submitted = st.form_submit_button("Ask")

    if submitted and question:
        with st.spinner("Thinking..."):
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 10})
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

else:
    st.info("👈 Upload a PDF from the sidebar to get started!")