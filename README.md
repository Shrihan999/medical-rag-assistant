# 🩺 Medical RAG Assistant – Healthcare QA System

## 📌 Overview
The **Medical RAG Assistant** is a domain-specific Question Answering (QA) system designed to provide **accurate, trustworthy, and explainable** answers to medical queries.  
It leverages a **local Large Language Model (LLM)** combined with a **vector-based retrieval system** to ground responses in **verified medical sources** — reducing hallucinations and misinformation often found in general-purpose AI.

---

## 🚀 Problem Statement
Accessing **reliable** and **structured** medical information is challenging for both patients and healthcare professionals.  
General AI models may produce **misleading** or **hallucinated** answers when asked medical questions.

**Goal:**  
To build a **domain-specific AI assistant** that:
- Uses **verified and up-to-date medical datasets**.
- Provides **explainable** and **context-grounded** answers.
- Runs locally for **data privacy** and **fast response times**.

---

## 💡 Proposed Solution
We implemented a **Retrieval-Augmented Generation (RAG)** pipeline that:
1. Loads and preprocesses a **Medical QA dataset** from Hugging Face.
2. Ingests **the latest Clinical Practice Guidelines (CPGs)**, **recent medical journals**, and **peer-reviewed research papers** in PDF format.
3. Generates **semantic embeddings** using `SentenceTransformer`.
4. Stores embeddings in a **FAISS index** for fast vector similarity search.
5. Retrieves relevant context for a query.
6. Passes context + query to a **local LLM** (`Med-Qwen2-7B-GGUF`) for answer generation.
7. Delivers the output via a **Streamlit web interface**.

---

## 🛠 Tech Stack
- **Datasets & Sources**:
  - Hugging Face Medical QA Dataset
  - Latest Clinical Practice Guidelines (CPGs)
  - Recent medical journals & research papers (PDF format)
- **Embedding Model**: [SentenceTransformers](https://www.sbert.net/)
- **Vector Store**: [FAISS](https://faiss.ai/)
- **LLM**: Med-Qwen2-7B-GGUF (runs locally with `llama-cpp` or `transformers`)
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python (`pandas`, `transformers`, `sentence-transformers`, `faiss`, `pypdf`)

---

## 🔄 Workflow
1. **Data Acquisition** – Load Hugging Face dataset and ingest PDFs containing latest CPGs, journals, and research papers.
2. **Data Preprocessing** – Clean & format text into structured documents with metadata.
3. **Embedding Generation** – Convert text into dense vectors using SentenceTransformer.
4. **Index Creation** – Store embeddings in FAISS for semantic search.
5. **Query Processing** – Retrieve top-k relevant documents for a user query.
6. **Answer Generation** – Pass query + context to LLM for final response.
7. **User Interface** – Display answers with supporting context on Streamlit.

---

## 📂 System Architecture
**Layers**:
- **Data Layer** – Medical QA dataset + latest CPGs, journals, and research papers.
- **Embedding & Retrieval Layer** – SentenceTransformer + FAISS.
- **LLM Layer** – Med-Qwen2-7B-GGUF.
- **Application Layer** – Streamlit frontend with adjustable parameters (Top-k, temperature, max tokens).

---

## 📊 Results
- Processed **50,000+ medical Q&A pairs** plus multiple latest CPGs and research publications.
- Delivered **real-time, context-grounded answers** based on up-to-date medical knowledge.
- Provided **explainable and relevant** responses for a wide range of medical questions.

---



## 📦 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/<shrihan999>/medical-rag-assistant.git
cd medical-rag-assistant

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## 📜 License
This project is licensed under the **MIT License** — feel free to use and modify for your own work.
