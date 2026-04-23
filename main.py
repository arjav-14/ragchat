import os
import time
import json
import logging
import tempfile
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# LangChain loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# =========================
# CONFIG & LOGGING
# =========================

load_dotenv(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = "llama-3.3-70b-versatile"
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K         = 3
MAX_TOKENS    = 600
TEMPERATURE   = 0.2

SUPPORTED_TYPES = ["pdf", "docx", "txt", "md", "csv", "xlsx", "xls", "pptx", "json"]

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Multi-File RAG Chatbot",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #0e1117; color: #e0e0e0; }

    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    .chat-user {
        background: #1c2a3a;
        border-left: 3px solid #58a6ff;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 0.95rem;
    }
    .chat-bot {
        background: #161b22;
        border-left: 3px solid #3fb950;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .chat-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 6px;
        opacity: 0.6;
    }

    .stTextInput > div > div > input {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e0e0e0 !important;
        border-radius: 8px !important;
    }

    .stButton > button {
        background-color: #238636 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
        padding: 8px 20px !important;
    }
    .stButton > button:hover { background-color: #2ea043 !important; }

    [data-testid="stFileUploader"] {
        background: #161b22;
        border: 1px dashed #30363d;
        border-radius: 8px;
        padding: 8px;
    }

    .file-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.72rem;
        font-family: 'IBM Plex Mono', monospace;
        margin: 2px 3px;
    }
    .tag-pdf   { background:#2d1b1b; color:#f47174; border:1px solid #8b3a3a; }
    .tag-docx  { background:#1b2a3d; color:#58a6ff; border:1px solid #1f6feb; }
    .tag-txt   { background:#1e2d1e; color:#6bd186; border:1px solid #238636; }
    .tag-csv   { background:#2d2a1b; color:#e3b341; border:1px solid #9e6a03; }
    .tag-xlsx  { background:#1b2d1e; color:#3fb950; border:1px solid #196c2e; }
    .tag-pptx  { background:#2d1f1b; color:#f0883e; border:1px solid #9b4a1a; }
    .tag-json  { background:#2a1b2d; color:#d2a8ff; border:1px solid #6e40c9; }
    .tag-md    { background:#1b2a2d; color:#76e3ea; border:1px solid #1b6970; }
    .tag-other { background:#21262d; color:#8b949e; border:1px solid #30363d; }

    .source-box {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.8rem;
        color: #8b949e;
        margin-top: 4px;
        font-family: 'IBM Plex Mono', monospace;
        line-height: 1.6;
    }

    .app-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem;
        font-weight: 600;
        color: #e0e0e0;
        margin-bottom: 4px;
    }
    .app-subheader { font-size: 0.82rem; color: #8b949e; margin-bottom: 20px; }

    hr { border-color: #21262d !important; }
    .stSpinner > div { border-top-color: #58a6ff !important; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# =========================
# SESSION STATE
# =========================

def init_session():
    defaults = {
        "chat_history": [],
        "vectorstore": None,
        "loaded_files": [],
        "total_chunks": 0,
        "groq_client": None,
        "api_key": GROQ_API_KEY,
        "error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# =========================
# FILE TYPE HELPERS
# =========================

def get_file_ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else "other"

def get_tag_class(ext: str) -> str:
    mapping = {
        "pdf": "tag-pdf", "docx": "tag-docx", "doc": "tag-docx",
        "txt": "tag-txt", "md": "tag-md",
        "csv": "tag-csv", "xlsx": "tag-xlsx", "xls": "tag-xlsx",
        "pptx": "tag-pptx", "ppt": "tag-pptx",
        "json": "tag-json",
    }
    return mapping.get(ext, "tag-other")

def file_icon(ext: str) -> str:
    icons = {
        "pdf": "📄", "docx": "📝", "doc": "📝",
        "txt": "📃", "md": "📋",
        "csv": "📊", "xlsx": "📊", "xls": "📊",
        "pptx": "📑", "ppt": "📑",
        "json": "🔧",
    }
    return icons.get(ext, "📁")


# =========================
# LOADERS
# =========================

def load_file(uploaded_file) -> list:
    ext = get_file_ext(uploaded_file.name)
    suffix = f".{ext}"
    tmp_path = None
    docs = []

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if ext == "pdf":
            docs = PyPDFLoader(tmp_path).load()

        elif ext in ("docx", "doc"):
            docs = Docx2txtLoader(tmp_path).load()

        elif ext in ("txt", "md"):
            docs = TextLoader(tmp_path, encoding="utf-8").load()

        elif ext == "csv":
            docs = CSVLoader(tmp_path, encoding="utf-8").load()

        elif ext in ("xlsx", "xls"):
            docs = UnstructuredExcelLoader(tmp_path).load()

        elif ext in ("pptx", "ppt"):
            docs = UnstructuredPowerPointLoader(tmp_path).load()

        elif ext == "json":
            with open(tmp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            text = json.dumps(data, indent=2)
            docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]

        else:
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]

        for doc in docs:
            doc.metadata["filename"] = uploaded_file.name

        logger.info(f"Loaded {uploaded_file.name} -> {len(docs)} doc(s)")
        return docs

    except Exception as e:
        logger.error(f"Error loading {uploaded_file.name}: {e}")
        return []

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# =========================
# EMBEDDING & INDEXING
# =========================

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_vectorstore(all_docs: list):
    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(all_docs)
    if not chunks:
        return None, 0
    embeddings = load_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    return vs, len(chunks)


# =========================
# GROQ LLM
# =========================

def get_groq_client():
    if st.session_state.groq_client is None:
        key = st.session_state.api_key
        if not key:
            return None
        st.session_state.groq_client = Groq(api_key=key)
    return st.session_state.groq_client


def ask_llm(query: str, context: str) -> str:
    client = get_groq_client()
    if client is None:
        return "Groq API key not set. Please enter it in the sidebar."

    prompt = f"""Answer the question using ONLY the context below.
Be detailed and structured. Use plain text only, no markdown symbols like **, ##, or *.
Use numbered points when listing items.

Context:
{context}

Question: {query}

Answer:"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant answering questions from uploaded documents. "
                        "Be detailed and structured. Use plain text only, no **, no ##, no *. "
                        "Use numbered lists when listing points."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return f"Error from Groq API: {str(e)}"


def retrieve_context(query: str):
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content[:600] for doc in docs])
    sources = [
        {
            "file": doc.metadata.get("filename", doc.metadata.get("source", "?")),
            "page": doc.metadata.get("page", ""),
            "snippet": doc.page_content[:120].replace("\n", " ") + "..."
        }
        for doc in docs
    ]
    return context, sources


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.markdown('<div class="app-header">🗂️ RAG Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subheader">Chat with any file type</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Groq API Key**")
    api_input = st.text_input(
        "API Key", value=st.session_state.api_key,
        type="password", placeholder="gsk_...",
        label_visibility="collapsed"
    )
    if api_input != st.session_state.api_key:
        st.session_state.api_key = api_input
        st.session_state.groq_client = None

    st.markdown("---")

    st.markdown("**Upload Files**")
    st.markdown(
        f'<div style="font-size:0.75rem;color:#8b949e;margin-bottom:6px;">Supported: {", ".join(SUPPORTED_TYPES)}</div>',
        unsafe_allow_html=True
    )
    uploaded_files = st.file_uploader(
        "Upload files",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        current_names = sorted([f.name for f in uploaded_files])
        if current_names != sorted(st.session_state.loaded_files):
            with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
                all_docs = []
                failed = []
                for uf in uploaded_files:
                    docs = load_file(uf)
                    if docs:
                        all_docs.extend(docs)
                    else:
                        failed.append(uf.name)

                if all_docs:
                    vs, n_chunks = build_vectorstore(all_docs)
                    st.session_state.vectorstore = vs
                    st.session_state.loaded_files = current_names
                    st.session_state.total_chunks = n_chunks
                    st.session_state.chat_history = []
                    st.session_state.error = None
                    if failed:
                        st.warning(f"Could not load: {', '.join(failed)}")
                    else:
                        st.success(f"{len(uploaded_files)} file(s) ready!")
                else:
                    st.error("Failed to extract text from all files.")

    if st.session_state.loaded_files:
        st.markdown("---")
        st.markdown("**Loaded Files**")
        for fname in st.session_state.loaded_files:
            ext = get_file_ext(fname)
            tag_cls = get_tag_class(ext)
            icon = file_icon(ext)
            short = fname[:22] + "..." if len(fname) > 25 else fname
            st.markdown(
                f'<span class="file-tag {tag_cls}">{icon} {short}</span>',
                unsafe_allow_html=True
            )
        st.markdown(
            f'<div style="font-size:0.78rem;color:#8b949e;margin-top:8px;font-family:IBM Plex Mono,monospace;">'
            f'Files: {len(st.session_state.loaded_files)}<br>'
            f'Chunks: {st.session_state.total_chunks}<br>'
            f'Model: {GROQ_MODEL[:22]}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset All"):
            st.session_state.chat_history = []
            st.session_state.vectorstore = None
            st.session_state.loaded_files = []
            st.session_state.total_chunks = 0
            st.rerun()

    st.markdown("""
    <div style="font-size:0.72rem;color:#484f58;margin-top:16px;font-family:IBM Plex Mono,monospace;">
    Embeddings: BAAI/bge-small-en-v1.5<br>
    Vector DB: FAISS (in-memory)<br>
    LLM: Groq Cloud
    </div>
    """, unsafe_allow_html=True)


# =========================
# MAIN CHAT AREA
# =========================

st.markdown("### Chat")

if not st.session_state.api_key:
    st.warning("Enter your Groq API key in the sidebar to get started.")
elif st.session_state.vectorstore is None:
    st.info("Upload one or more files from the sidebar to begin chatting.")
else:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-user">
                <div class="chat-label">You</div>
                {msg["content"]}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-bot">
                <div class="chat-label">Assistant</div>
                {msg["content"]}
            </div>""", unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("View sources", expanded=False):
                    for s in msg["sources"]:
                        page_info = f" — Page {int(s['page'])+1}" if s.get("page") != "" else ""
                        st.markdown(
                            f'<div class="source-box">'
                            f'{file_icon(get_file_ext(s["file"]))} {s["file"]}{page_info}<br>'
                            f'{s["snippet"]}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Message", placeholder="Ask anything about your files...",
            label_visibility="collapsed", key="query_input"
        )
    with col2:
        send = st.button("Send →")

    if send and query.strip():
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.spinner("Thinking..."):
            start = time.time()
            context, sources = retrieve_context(query)
            answer = ask_llm(query, context)
            elapsed = round(time.time() - start, 2)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "time": elapsed
        })
        logger.info(f"Query answered in {elapsed}s")
        st.rerun()
