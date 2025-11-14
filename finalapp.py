import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

# Streamlit Page Configuration
st.set_page_config(
    page_title="NVIDIA NIM PDF Chatbot",
    page_icon="🧠",
    layout="wide",
)

from dotenv import load_dotenv
load_dotenv()

## Loading Nvidia API Key
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

# Initializing the LLM Model from NVIDIA NIM
llm = ChatNVIDIA(
    model_name = "meta/llama-3.3-70b-instruct"
    )

# Default folder with PDFs
DOCUMENT_DIR = "./us_census"  


# Data Ingestion, Vector Embedding, and Vector DB
def vector_embedding():
    if "vectors" in st.session_state:
        return

    with st.spinner("📚 Loading PDFs and building FAISS vector store..."):
        loader = PyPDFDirectoryLoader(DOCUMENT_DIR)
        docs = loader.load()

        if not docs:
            st.error(
                f"No PDFs found in `{DOCUMENT_DIR}`.\n"
                f"Add some PDF files to that folder and restart the app."
            )
            st.stop()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=50,
        )
        # limit to first 30 docs to keep it fast
        final_documents = text_splitter.split_documents(docs[:30])
        embeddings = NVIDIAEmbeddings()
        vectors = FAISS.from_documents(final_documents, embeddings)

        # Storing in session_state
        st.session_state.embeddings = embeddings
        st.session_state.loader = loader
        st.session_state.docs = docs
        st.session_state.text_splitter = text_splitter
        st.session_state.final_documents = final_documents
        st.session_state.vectors = vectors
        st.session_state.num_docs = len(docs)
        st.session_state.num_chunks = len(final_documents)

    st.success(
        f"✅ Vector store ready! "
        f"{st.session_state.num_docs} PDFs → {st.session_state.num_chunks} chunks."
    )



# Streamlit Sidebar Section
st.sidebar.title("ℹ️ About this app")
st.sidebar.markdown(
    """
Chat with your **own PDF documents** using:

- **NVIDIA NIM** (`meta/llama-3.3-70b-instruct`)
- **FAISS** vector store
- **LangChain** for retrieval
- **Streamlit** for the UI

Place your PDFs inside the `./us_census` folder to get started.
"""
)

top_k = st.sidebar.slider(
    "Number of chunks to retrieve (k)",
    min_value=2,
    max_value=8,
    value=4,
    step=1,
    help="How many most similar text chunks to send to the model.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("Made using NVIDIA NIM + LangChain + Streamlit")



# Streamlit Main Section
st.title("🧠 NVIDIA NIM PDF Chatbot")
st.caption("Ask questions grounded in your local PDF knowledge base.")

# Metrics row
col1, col2, col3 = st.columns(3)
col1.metric("PDF files loaded", st.session_state.get("num_docs", 0))
col2.metric("Text chunks", st.session_state.get("num_chunks", 0))
if "last_response_time" in st.session_state:
    col3.metric(
        "Last response (s)",
        f"{st.session_state['last_response_time']:.2f}",
    )
else:
    col3.metric("Last response (s)", "—")

# Button to manually build embeddings (optional)
if st.button("📦 Build / Refresh Vector Store"):
    vector_embedding()

# -----------------------------
# Prompt template
# -----------------------------
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant that only answers using the provided context.

Guidelines:
- If the answer is not in the context, say you don't know.
- Do NOT hallucinate or invent numbers.
- Be concise and well structured.

<context>
{context}
</context>

Question: {input}
"""
)

user_question = st.text_input("🔎 Enter your question about the documents")


# Handling User Input
if user_question:
    if "vectors" not in st.session_state:
        vector_embedding()

    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever(
            search_kwargs={"k": top_k}
        )
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner("🤖 Thinking with NVIDIA NIM..."):
            start = time.perf_counter()
            response = retrieval_chain.invoke({"input": user_question})
            elapsed = time.perf_counter() - start

        st.session_state["last_response_time"] = elapsed
        st.subheader("✅ Answer")
        st.write(response["answer"])
        st.caption(f"Response time: {elapsed:.2f} seconds")

        # Showing Source Chunks
        with st.expander("📎 Document similarity search (source chunks)"):
            for i, doc in enumerate(response["context"], start=1):
                st.markdown(f"**Chunk #{i}**")
                st.write(doc.page_content)
                meta = []
                if "source" in doc.metadata:
                    meta.append(f"Source: `{doc.metadata['source']}`")
                if "page" in doc.metadata:
                    meta.append(f"Page: {doc.metadata['page']}`")
                if meta:
                    st.caption(" | ".join(meta))
                st.write("---")
    else:
        st.error("Vector store is not initialized. Please try again.")