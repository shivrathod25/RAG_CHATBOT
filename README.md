# RAG Chatbot with LangGraph, ChromaDB, and Google Gemini

An intelligent Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangGraph**, **Chroma DB**, and **Google Gemini AI**. This application indexes PDF documents into a vector database, processes context using advanced graph-based workflows, and provides an interactive web chat UI.

---

## 🌟 Features

- 📄 **PDF Document Ingestion**: Upload and index multiple PDF files dynamically into vector embeddings.
- ⚡ **Vector Search**: High-performance semantic search using Chroma DB embeddings (`models/text-embedding-004`).
- 🤖 **LangGraph Workflow**: StateGraph architecture for intelligent routing, document retrieval, context augmentation, and generation.
- 🔄 **Automatic Model Fallback**: Resilience against API capacity issues (503 / 429) across Gemini 1.5 & 2.0 Flash models.
- 💻 **Modern Web Frontend**: Clean, responsive frontend interface (`index.html`) communicating with backend REST endpoints via CORS.

---

## 📁 Repository Structure

```
RAG_Chatbot/
├── backend/
│   ├── app.py             # FastAPI server with LangGraph & ChromaDB pipeline
│   └── RagChatbot.ipynb   # Interactive Jupyter Notebook for RAG experimentation
├── frontend/
│   └── index.html         # Responsive web UI frontend
├── HR_Policy.pdf          # Sample document
├── IT_Security_Policy.pdf # Sample document
├── ML.pdf                 # Sample document
├── .env.example           # Environment variable template
├── .gitignore             # Git ignore file
├── requirements.txt       # Python dependency list
├── HOW_TO_RUN.txt         # Quick start reference guide
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** installed
- **Google Gemini API Key** (obtainable from [Google AI Studio](https://aistudio.google.com/))

---

### 📥 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/shivrathod25/RAG_Chatbot.git
cd RAG_Chatbot
pip install -r requirements.txt
```

---

### 🔑 2. Environment Configuration

Copy `.env.example` to `.env` and insert your Google Gemini API key:

```bash
# On Linux/macOS
cp .env.example .env

# On Windows PowerShell
copy .env.example .env
```

Open `.env` and set your key:
```env
GOOGLE_API_KEY="your_actual_gemini_api_key_here"
```

---

### 🏃 3. Running the Application

#### Step 1: Launch Backend API
Start the FastAPI server from the project root directory:

```bash
python backend/app.py
```
*The backend API will start on `http://127.0.0.1:8000`.*

#### Step 2: Launch Frontend Interface
Open `frontend/index.html` in your web browser, or serve it using Python's simple HTTP server:

```bash
python -m http.server 3000 --directory frontend
```
*Access the interface at `http://localhost:3000`.*

---

## 🔌 API Endpoints

- `GET /`: Health check / API status
- `POST /ask`: Query the RAG Chatbot (`{"question": "What is the security policy?"}`)
- `POST /upload`: Upload and ingest new PDF documents into Chroma DB

---

## 🛡️ Security Note

Make sure **never** to commit your actual `.env` file containing API keys to GitHub. The `.gitignore` file included in this repository prevents sensitive files from being pushed.
