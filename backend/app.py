import os
import sys
import time
from typing import List, Dict, Any, TypedDict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END

# Telemetry control
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["CHROMA_TELEMETRY"] = "False"

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load Environment Variables (.env)
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv()

# Normalize Google Gemini API Key
api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key

# Configure standard I/O encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Initialize FastAPI App
app = FastAPI(
    title="Enterprise RAG Chatbot API",
    description="Backend API powered by LangGraph, Chroma DB, and Google Gemini",
    version="1.0.0"
)

# Configure CORS for Frontend Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static directory if exists
frontend_dir = os.path.join(BASE_DIR, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# LLM Fallback Models (Handling 503 capacity & 429 quota limits across regions)
FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite"
]

def invoke_llm_with_fallback(prompt_text: str):
    """Invoke Gemini LLM with automatic fallback across models if 503 capacity or 429 quota errors occur."""
    last_error = None
    for model_name in FALLBACK_MODELS:
        try:
            model_llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
            return model_llm.invoke(prompt_text)
        except Exception as e:
            last_error = e
            err_msg = str(e).encode('ascii', 'backslashreplace').decode('ascii')
            print(f"Model '{model_name}' temporary notice: {err_msg}. Trying next fallback...")
            time.sleep(0.3)
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("All LLM fallback models failed.")

# Vector DB & Knowledge Base Setup
COLLECTION_NAME = "chroma_db"
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_db")
PDF_FILES = ["ML.pdf", "HR_Policy.pdf", "IT_Security_Policy.pdf"]

_vectorstore = None
_retriever = None

def get_embeddings():
    """Retrieve embeddings instance."""
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

def initialize_vectorstore():
    """Initialize Chroma vector store. Ensures all PDF documents are loaded and indexed."""
    global _vectorstore, _retriever
    if _vectorstore is not None:
        return _vectorstore

    embeddings = get_embeddings()
    rebuild_needed = False
    
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        try:
            vs = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
            # Verify if documents from all PDF files exist in the collection
            existing_data = vs.get()
            existing_sources = set()
            if existing_data and "metadatas" in existing_data and existing_data["metadatas"]:
                for meta in existing_data["metadatas"]:
                    if meta and "source" in meta:
                        existing_sources.add(meta["source"])
            
            missing_files = [f for f in PDF_FILES if f not in existing_sources]
            if not missing_files and len(existing_data.get("ids", [])) > 0:
                print(f"[INFO] Loaded existing Chroma DB from '{PERSIST_DIRECTORY}' with sources: {existing_sources}")
                _vectorstore = vs
                _retriever = vs.as_retriever(search_kwargs={"k": 6})
                return _vectorstore
            else:
                print(f"[INFO] Chroma DB missing sources {missing_files}. Rebuilding...")
                rebuild_needed = True
        except Exception as e:
            print(f"[WARNING] Error checking existing Chroma DB ({e}). Rebuilding...")
            rebuild_needed = True

    print("[INFO] Building new Chroma DB from PDF documents...")
    all_documents = []
    for pdf_file in PDF_FILES:
        pdf_path = os.path.join(BASE_DIR, pdf_file)
        if os.path.exists(pdf_path):
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = pdf_file
            all_documents.extend(docs)
            print(f"Loaded {len(docs)} pages from {pdf_file}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"[INFO] Created {len(chunks)} text chunks.")

    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY
    )
    _retriever = _vectorstore.as_retriever(search_kwargs={"k": 6})
    return _vectorstore

def get_retriever():
    """Lazy retriever access."""
    global _retriever
    if _retriever is None:
        initialize_vectorstore()
    return _retriever

# LangGraph State Graph Definition
class GraphState(TypedDict):
    question: str
    documents: List[Any]
    answer: str

def retrieve_node(state: GraphState):
    question = state["question"]
    retriever_instance = get_retriever()
    
    # 1. Primary vector search
    documents = retriever_instance.invoke(question)
    
    # 2. Query Expansion for abbreviations and Hinglish terms
    expanded_queries = []
    lower_q = question.lower()
    
    # Abbreviation expansion
    if "ml" in lower_q.split() or "ml" in lower_q:
        expanded_queries.append(lower_q.replace("ml", "machine learning"))
    if "hr" in lower_q.split():
        expanded_queries.append(lower_q.replace("hr", "human resources"))
    if "it" in lower_q.split():
        expanded_queries.append(lower_q.replace("it", "information technology"))
        
    # Hinglish query cleanup
    cleaned_query = lower_q
    for phrase in ["kya hai", "kaise kaam karta hai", "kaise hota hai", "kya hota hai", "batao", "bataiye"]:
        cleaned_query = cleaned_query.replace(phrase, "").strip()
    if cleaned_query and cleaned_query != lower_q:
        expanded_queries.append(cleaned_query)

    # Execute expanded queries if needed to supplement documents
    existing_contents = {doc.page_content for doc in documents}
    for eq in expanded_queries:
        if eq and eq != lower_q:
            extra_docs = retriever_instance.invoke(eq)
            for doc in extra_docs:
                if doc.page_content not in existing_contents:
                    documents.append(doc)
                    existing_contents.add(doc.page_content)

    return {"documents": documents}

def generate_node(state: GraphState):
    question = state["question"]
    documents = state["documents"]

    context = "\n\n".join(doc.page_content for doc in documents)
    
    prompt = f"""You are a professional enterprise RAG AI Assistant for HR Policy, IT Security Policy, and Machine Learning Fundamentals.

Context from Documents:
{context}

User Question:
{question}

Instructions:
1. Conversational Greetings & General Queries:
   If the user asks a basic conversational or greeting question (such as "Hi", "Hello", "Hey", "How are you?", "Who are you?", "Good morning", "Help", etc.), respond politely, warmly, and naturally as an AI assistant. State that you are here to help with questions about HR Policies, IT Security Guidelines, or Machine Learning Fundamentals. Do NOT state that information is unavailable for basic greetings or chit-chat.

2. Specific Knowledge / Document Questions (Strict Grounding):
   If the user asks a factual or domain question that requires specific information:
   - Answer accurately using ONLY the information provided in the Context from Documents above.
   - If the requested specific information is NOT present or cannot be inferred from the provided context, respond ONLY with:
     "The requested information could not be found in the provided documents."
   - Do NOT invent, make up, or hallucinate any facts outside the provided document context.

3. Language Support:
   Support questions in English or Hinglish (Hindi written in Roman script). Respond in the same language/style as the user's question.

4. Formatting:
   Format responses with clear markdown, incorporating a short intro and bullet points where applicable."""

    response = invoke_llm_with_fallback(prompt)
    content = response.content
    if isinstance(content, list):
        answer_text = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
    else:
        answer_text = content

    return {"answer": answer_text}

# Build LangGraph App Workflow
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

rag_app = workflow.compile()

# API Pydantic Schemas
class QueryRequest(BaseModel):
    question: str

class DocumentChunk(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    question: str
    answer: str
    documents: List[DocumentChunk]

# REST Endpoints
@app.get("/")
def read_root():
    """Serve the interactive web frontend at root URL if available, else API status."""
    frontend_path = os.path.join(BASE_DIR, "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path, media_type="text/html")
    return {
        "status": "online",
        "service": "Enterprise RAG Chatbot API",
        "endpoints": {
            "health": "/api/health",
            "chat": "/api/chat",
            "documents": "/api/documents",
            "docs": "/docs"
        }
    }

@app.get("/favicon.ico", include_in_schema=False)
def get_favicon():
    """Silence browser favicon requests gracefully."""
    return Response(status_code=204)

@app.get("/api/health")
def health_check():
    """Health check endpoint for Render monitoring and client status checks."""
    has_key = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    return {
        "status": "healthy" if has_key else "needs_configuration",
        "service": "RAG Chatbot API",
        "api_key_configured": has_key,
        "knowledge_base": PDF_FILES
    }

@app.get("/api/documents")
def get_documents_info():
    """Return list of knowledge base documents with metadata."""
    return {
        "documents": [
            {"filename": "HR_Policy.pdf", "title": "HR Policy Manual", "category": "HR Policy & Benefits", "pages": 14},
            {"filename": "IT_Security_Policy.pdf", "title": "IT & Data Security Policy", "category": "IT & Cyber Security", "pages": 17},
            {"filename": "ML.pdf", "title": "Machine Learning & AI Notes", "category": "Machine Learning Fundamentals", "pages": 9}
        ]
    }

@app.get("/pdfs/{filename}")
def get_pdf(filename: str):
    """Serve PDF documents for frontend viewer/download."""
    if filename not in PDF_FILES:
        raise HTTPException(status_code=404, detail="PDF file not found in knowledge base.")
    file_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' does not exist on server.")
    return FileResponse(file_path, media_type="application/pdf", filename=filename)

@app.post("/api/chat", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """Execute LangGraph RAG workflow on incoming question."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY is not configured. Please add your GOOGLE_API_KEY in the Render Environment Variables dashboard."
        )

    try:
        result = rag_app.invoke({
            "question": request.question,
            "documents": [],
            "answer": ""
        })
        
        formatted_docs = []
        for doc in result.get("documents", []):
            formatted_docs.append(DocumentChunk(
                page_content=doc.page_content,
                metadata=doc.metadata
            ))

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            documents=formatted_docs
        )
    except Exception as e:
        print(f"Error in query_rag: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def free_port_if_occupied(port: int = 8000):
    """Pre-flight check to detect and automatically release port if an orphaned process is listening on it."""
    import socket, subprocess
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_busy = sock.connect_ex(('127.0.0.1', port)) == 0
    sock.close()
    
    if is_busy:
        print(f"[WARNING] Port {port} is currently occupied by another process. Attempting automatic cleanup...")
        try:
            if os.name == 'nt':
                # Windows: Find and terminate process occupying target port
                ps_cmd = f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | ForEach-Object {{ Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }}"
                subprocess.run(["powershell", "-Command", ps_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                # Unix/Linux/macOS
                subprocess.run(["fuser", "-k", f"{port}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1)
            print(f"[SUCCESS] Port {port} successfully freed.")
        except Exception as e:
            print(f"[WARNING] Could not automatically free port {port}: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    
    if host == "127.0.0.1":
        free_port_if_occupied(port)
        
    print(f"Starting FastAPI server on {host}:{port}...")
    uvicorn.run(app, host=host, port=port)
