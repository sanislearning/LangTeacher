# 🧠 LangTeacher – Learn from Your PDFs with AI

LangTeacher is an AI-powered tutor that helps you interactively learn from any PDF. Just upload a document and start asking questions — your personal AI tutor will guide you with clear, simple answers based on the file content.

[Try the LangTeacher App](https://langteacher.streamlit.app/)
Built using [LangChain](https://www.langchain.com/), [Google Gemini](https://ai.google.dev/), and [Streamlit](https://streamlit.io/).

---

## 🚀 Features

- 📄 Upload any PDF (e.g., lecture notes, reports, articles)
- 🤖 Ask questions and receive context-aware answers
- 🧩 Chunking and semantic search via FAISS & HuggingFace embeddings
- 🗣️ Conversational memory for follow-up questions
- 🧼 Clean, Streamlit-powered chat UI

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM:** Gemini 2.0 Flash (`langchain-google-genai`)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Retrieval:** FAISS (via `langchain_community`)
- **Chunking:** RecursiveCharacterTextSplitter
- **PDF Parsing:** LangChain's `PyPDFLoader`

---

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/sanislearning/LangTeacher
cd langteacher
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ Make sure you're using Python 3.10 or above.

### 3. Add your Gemini API key

Create a `.env` file in the root directory:

```
GOOGLE_API_KEY=your_api_key_here
```

---

## ▶️ Running the App

```bash
streamlit run main.py
```

Open your browser at [http://localhost:8501](http://localhost:8501), upload a PDF, and start learning!

---

## 🧠 Example Use Cases

* Study from your own notes or books
* Understand research papers
* Ask questions from technical manuals or company docs

---

## 🤝 Contributing

Pull requests and suggestions are welcome! Feel free to open an issue or submit a PR.

---
