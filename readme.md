# 📄 GenAI Document Intelligence (RAG)

## 📌 Project Overview

**GenAI Document Intelligence (RAG)** is an **offline Retrieval-Augmented Generation (RAG) application** that enables users to ask natural-language questions and receive **concise, accurate answers** directly from unstructured business documents.

The system is designed to handle **financial reports, shareholder letters, and enterprise documents**, making it suitable for real-world use cases such as:

* Financial analysis
* Policy and compliance search
* Business intelligence
* Enterprise knowledge assistants

The application uses **semantic search + large language models** to ensure answers are grounded in the source documents and includes **source attribution** for transparency.

## 🎯 Problem Statement

Traditional keyword-based document search:

* Misses semantic meaning
* Returns large, unstructured text blocks
* Requires manual reading

This project solves those limitations by:

* Converting documents into embeddings
* Retrieving only the most relevant context
* Generating **short, accurate, context-aware answers**

## 🚀 Key Features

* 🔍 **Semantic Search** using FAISS vector database
* 🧠 **Retrieval-Augmented Generation (RAG)** pipeline
* ⚡ **Fast retrieval** with MMR (Maximal Marginal Relevance)
* 📄 **DOCX document ingestion**
* 📴 **Fully offline** (no OpenAI / API keys required)
* 🧾 **Source attribution** for every answer
* ✂️ **Concise 2-line detailed answers**
* 🖥️ **Interactive Gradio UI**
* ♻️ **Answer caching** for instant repeated queries

## 🏗️ System Architecture

```
DOCX Documents
      ↓
Document Loader (Docx2txt)
      ↓
Text Chunking (RecursiveCharacterTextSplitter)
      ↓
Sentence Embeddings (MiniLM)
      ↓
FAISS Vector Store
      ↓
MMR Retriever
      ↓
Prompt + LLM (FLAN-T5)
      ↓
Concise Answer + Sources
```

## 📁 Project Structure

```
project/
│
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── faiss_index/           # Saved FAISS vector index
│
└── data/                  # Input documents
    ├── 2022_Annual_Report.docx
    ├── 2022_Shareholder_Letter.docx
    └── MSFT_FY22Q4_10K.docx
```

## ⚙️ Technology Stack

| Component  | Technology                     |
| ---------- | ------------------------------ |
| UI         | Gradio                         |
| LLM        | Hugging Face FLAN-T5           |
| Embeddings | Sentence-Transformers (MiniLM) |
| Vector DB  | FAISS                          |
| Framework  | LangChain (Runnable APIs)      |
| Language   | Python                         |
| Deployment | Local / Hugging Face Spaces    |

## 🧠 Core Design Decisions

### 1️⃣ Retrieval-Augmented Generation (RAG)

Instead of asking the LLM directly, the system:

* Retrieves relevant document chunks
* Injects them into the prompt
* Ensures answers are grounded in data

This **reduces hallucinations** and improves trust.

### 2️⃣ FAISS + MMR Retrieval

* FAISS provides **fast similarity search**
* MMR reduces redundant chunks
* Smaller context → faster inference → better answers

### 3️⃣ Prompt-Controlled Answer Length

Answers are constrained to:

* **Maximum 2 concise but informative lines**
* No unnecessary explanations
* Business-friendly output

### 4️⃣ Offline-First Design

* No API keys
* No external calls
* Safe for enterprise environments

### 5️⃣ Answer Caching

Repeated questions return **instant responses**, improving:

* Latency
* User experience
* System efficiency

## ▶️ Installation & Setup

### 1️⃣ Create Virtual Environment (Recommended)

```bash
conda create -n rag python=3.10 -y
conda activate rag
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

```bash
python app.py
```

The Gradio interface will open at:

```
http://localhost:7860
```

## 🧪 Example Questions

* What were Microsoft’s key revenue drivers in FY2022?
* How did cloud services perform in FY2022?
* What risks are mentioned in the FY22 10-K report?
* What message did leadership emphasize in the shareholder letter?

## 📊 Performance Optimizations

| Optimization      | Benefit                   |
| ----------------- | ------------------------- |
| MMR Retrieval     | Faster & diverse context  |
| Answer Cache      | Instant repeat responses  |
| Small LLM         | Reduced latency           |
| Chunking Strategy | Better retrieval accuracy |

## 🧠 Skills Demonstrated

* Retrieval-Augmented Generation (RAG)
* Vector databases (FAISS)
* Semantic search & embeddings
* Prompt engineering
* LLM inference optimization
* Gradio UI development
* Production-safe GenAI system design

## 📌 Project Description

**GenAI Document Intelligence (RAG)**
Developed an offline Retrieval-Augmented Generation system using LangChain, FAISS, and Hugging Face models to answer financial and business questions from unstructured documents. Implemented semantic search with MMR-based retrieval, optimized inference latency using caching, and deployed an interactive Gradio UI with concise, source-attributed responses.

## 🧪 Limitations & Future Improvements

* Supports DOCX files only (PDF support can be added)
* Single-user local deployment
* No chat history (can be added safely)

**Future Enhancements**

* PDF ingestion
* Chat-style UI
* Hybrid BM25 + vector retrieval
* API deployment (FastAPI)

## 📜 License

This project is intended for **educational and portfolio demonstration purposes**.

## ✅ Final Notes

This project reflects **real-world GenAI system design**, not just a demo:

* Stable
* Explainable
* Offline