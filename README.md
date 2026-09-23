# Enterprise RAG Knowledge Assistant & Chatbot

An intelligent, production-ready Retrieval-Augmented Generation (RAG) knowledge assistant built with **FastAPI**, **LangGraph**, **Chroma DB**, and **Google Gemini AI (Embeddings & LLMs)**.

---

## 🌟 Key Features

- 📄 **Multi-Domain Document Ingestion**: Ingests and semantically indexes PDF policies (**HR Policy**, **IT & Data Security Policy**, and **Machine Learning Notes**).
- 🧠 **Vector Indexing & Semantic Search**: Powered by Chroma DB and Google Gemini Embeddings (`models/gemini-embedding-001`).
- 🤖 **LangGraph Agentic Pipeline**: Cyclic StateGraph workflow orchestrating query expansion, vector search, and grounded response generation.
- 🛡️ **Zero Hallucination Grounding**: Strict guardrails preventing unsupported claims, with fallback mechanisms.
- ⚡ **Multi-Model LLM Resilience**: Automatic fallback across Gemini Flash models if 503 capacity or 429 quota limits occur.
- 🌐 **Unified Web UI & API**: Complete responsive interface served directly from the FastAPI backend or separately as a static app.
- 🚀 **Render 1-Click Ready**: Includes `render.yaml` Blueprint and `runtime.txt` for instant deployment.

---

## 📁 Repository Structure

```
RAG_Chatbot/
├── backend/
│   ├── app.py                  # FastAPI server with LangGraph & ChromaDB pipeline
│   └── RagChatbot.ipynb        # Interactive Jupyter Notebook for RAG experimentation
├── frontend/
│   └── index.html              # Responsive web chat UI
├── HR_Policy.pdf               # HR policy document
├── IT_Security_Policy.pdf      # IT security policy document
├── ML.pdf                      # Machine learning guide
├── render.yaml                 # Render Blueprint configuration (1-click deploy)
├── runtime.txt                 # Render Python runtime specification (python-3.11.9)
├── requirements.txt            # Python dependencies
├── RENDER_DEPLOYMENT_STEPS.txt # Detailed step-by-step Render deployment manual
├── HOW_TO_RUN.txt              # Quick start local run guide
└── README.md                   # Project documentation
```

---

## 🚀 Live Deployment on Render

### Option 1: Blueprint Deployment (Fastest)
1. Push this repository to your GitHub account: `https://github.com/shivrathod25/RAG_Chatbot`
2. Log in to [Render Dashboard](https://dashboard.render.com).
3. Click **New +** -> **Blueprint**.
4. Connect this repository. Render will automatically detect `render.yaml`.
5. Enter your **`GOOGLE_API_KEY`** in the prompt.
6. Click **Apply**. Render will build and launch your live application!

### Option 2: Standard Web Service Deployment
1. Click **New +** -> **Web Service** on Render.
2. Select your repository `shivrathod25/RAG_Chatbot`.
3. Configure:
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Free`
4. Under **Environment Variables**, add:
   - `GOOGLE_API_KEY`: `your_gemini_api_key`
   - `PYTHON_VERSION`: `3.11.9`
5. Click **Create Web Service**.

---

## 💻 Local Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Start Application
```bash
python backend/app.py
```
Open `http://localhost:8000` in your web browser to interact with the chatbot!

---

## 🔌 API Reference

- `GET /` - Interactive Web Chatbot UI
- `GET /api/health` - Service health and indexed documents status
- `GET /api/documents` - Metadata of knowledge base PDFs
- `GET /pdfs/{filename}` - Document viewer / download endpoint
- `POST /api/chat` - Query endpoint (`{"question": "What is the annual leave policy?"}`)
- `GET /docs` - Interactive Swagger API Documentation

---

## 📜 License
MIT
