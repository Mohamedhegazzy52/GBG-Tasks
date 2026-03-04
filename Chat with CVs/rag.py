import os
from dotenv import load_dotenv
from pathlib import Path


from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma , FAISS, Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# Load API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Folder that contains the CV files
DATA_DIR = Path(__file__).parent / "data"


# ── 1. Load all CVs from data/ folder ────────────────────────────────────────

def load_all_cvs():
    """Auto-load every PDF and TXT file found in the data/ folder."""
    docs = []
    files = list(DATA_DIR.glob("*.pdf")) + list(DATA_DIR.glob("*.txt"))

    if not files:
        raise FileNotFoundError(f"No CV files found in '{DATA_DIR}'. Add PDF or TXT files there.")

    for path in files:
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")
        docs.extend(loader.load())

    return docs, [f.name for f in files]


# ── 2. Document-Aware Chunking ────────────────────────────────────────────────

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    return splitter.split_documents(documents)


# ── 3. Embed & Store ──────────────────────────────────────────────────────────

def build_vectorstore(chunks):
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")#BAAI/bge-m3
   
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    
    return  vectorstore


# ── 4. MMR + Multi-Query Retriever ────────────────────────────────────────────

def build_retriever(vectorstore, llm):
    base_retriever = vectorstore.as_retriever(
        
        search_kwargs={"k": 3}
    )
    
    
    retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
    return  retriever


# ── 5. Prompt ─────────────────────────────────────────────────────────────────

PROMPT = PromptTemplate(
    template="""
You are an HR assistant. Use the context below (from loaded CVs) to answer the question.  

- If the answer is in the context → answer directly.  
- If not → rephrase the question to check for understanding.  
- If still no answer → respond with "I don't know".  
- Reply in the same language as the question.  

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"],
)


# ── 6. RAG Chain ──────────────────────────────────────────────────────────────

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever, llm):
    return (
        {"context": retriever | format_docs, "question": lambda x: x}
        | PROMPT
        | llm
        | StrOutputParser()
    )


# ── 7. Full Pipeline (called once at startup) ─────────────────────────────────

def build_pipeline():
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found.")

    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

    documents, file_names = load_all_cvs()
    chunks    = split_documents(documents)
    vs        = build_vectorstore(chunks)
    retriever = build_retriever(vs, llm)
    chain     = build_rag_chain(retriever, llm)

    return chain, file_names, len(chunks)