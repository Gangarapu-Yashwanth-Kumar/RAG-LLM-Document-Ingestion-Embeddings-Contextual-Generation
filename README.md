# 📚🤖 RAG + LLM: Document Ingestion, Embeddings & Contextual Generation

---

## 🔎 Overview
This repo demonstrates an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline:  
PDF 📄 ingestion → text chunking ✂️ → embeddings 🔗 → ChromaDB 📦 → retriever 🔍 → LLM 🤖 → contextual answers with sources.  
Built using **LangChain**, **HuggingFace**, **Groq LLMs**, and **Chroma**. 🚀  

---

## ⚙️ Setup

### Prerequisites
- Python 3.8+ 🐍  
- pip  
- (Optional) GPU for local inference  
- `poppler` installed for `unstructured[pdf]` parsing  

### Install Poppler
- Ubuntu/Debian: `sudo apt-get install -y poppler-utils`  
- macOS: `brew install poppler`  

### Create Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -U chromadb langchain langchain-groq langchain-community \
    langchain-chroma langchain-text-splitters transformers \
    sentence-transformers unstructured "unstructured[pdf]"
```
---

## 🧩 Components
- **Document Loader** — `UnstructuredFileLoader`  
- **Text Splitter** — `CharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=50`)  
- **Embeddings** — `HuggingFaceEmbeddings`  
- **Vector DB** — `Chroma` (`persist_directory="vector_db"`)  
- **Retriever** — `vectordb.as_retriever()`  
- **LLM** — `ChatGroq` (`llama-3.3-70b-versatile`)  
- **QA Chain** — `RetrievalQA.from_chain_type(...)`  
- **Utilities** — `requests` for fetching example PDFs  

---

## ✨ Features
- End-to-end workflow (ingest → retrieve → generate)  
- Persistent vector DB for re-use ⚡  
- HuggingFace embeddings for local inference 🧠  
- Real example dataset (IPL 2025 schedule) 🏏  
- RetrievalQA with source citations 📑  

---

## 📝 Responsibilities
- **UnstructuredFileLoader** → Parse PDFs into Document objects  
- **CharacterTextSplitter** → Break docs into manageable chunks  
- **HuggingFaceEmbeddings** → Convert text chunks → vectors  
- **Chroma** → Store & index vectors  
- **Retriever** → Fetch nearest neighbors for queries  
- **ChatGroq (LLM)** → Generate context-aware responses  
- **RetrievalQA** → Connect retriever + LLM + return answers + sources  

---

## 🛠️ Technologies Used
- Python 3.8+ 🐍  
- LangChain 🔗  
- `langchain-groq` (Groq LLM wrapper)  
- `langchain-chroma` & `chromadb`  
- `langchain-community`  
- Hugging Face sentence-transformers 🤗  
- `transformers`  
- `unstructured[pdf]` 📄  
- Poppler (system dependency)  
- Requests 🌐  

---

## 🏗️ How it works (Architecture)

flowchart LR

  A[📄 PDF / Documents] --> B[📥 UnstructuredFileLoader]
  
  B --> C[✂️ Text Splitter<br/>(chunk_size=1000,<br/>overlap=50)]
  
  C --> D[🔗 Embeddings<br/>(HuggingFaceEmbeddings)]
  
  D --> E[📦 Chroma Vector DB<br/>(persist_directory="vector_db")]
  
  E --> F[🔍 Retriever (as_retriever())]
  
  F --> G[⚙️ RetrievalQA Chain]
  
  H[🤖 LLM: ChatGroq<br/>(llama-3.3-70b-versatile)] --> G
  
  G --> I[✅ Answer + 📑 Source Documents]
  

- Flow summary: Ingest → Chunk → Embed → Index → Retrieve → Generate Answer
---
## 🚀 Quick Example (Usage)

```python
# 📥 Download example PDF (IPL 2025 Schedule)
import requests
url = "https://documents.iplt20.com/smart-images/..._Season_Schedule_2025.pdf"
resp = requests.get(url)
open("IPL_Schedule_2025.pdf", "wb").write(resp.content)

# 📄 Load document
from langchain_community.document_loaders import UnstructuredFileLoader
loader = UnstructuredFileLoader("IPL_Schedule_2025.pdf")
documents = loader.load()

# ✂️ Split into chunks
from langchain_text_splitters import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# 🔗 Generate embeddings + store in Chroma DB
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
embeddings = HuggingFaceEmbeddings()
vectordb = Chroma.from_documents(texts, embeddings, persist_directory="vector_db")

# 🔍 Retriever + ⚙️ QA Chain
retriever = vectordb.as_retriever()
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# ❓ Query the system
res = qa_chain({'query': 'Total runs scored by Virat Kohli'})
print(res['result'])
```
---
## 💡 Use Cases
- 📄 Ad-hoc Q&A over PDFs & reports  
- 👩‍💻 Lightweight support knowledge-bases  
- 📑 Research assistants w/ citations  
- 🏢 Enterprise documentation search  
- 🧪 Prototyping RAG before scaling to cloud vector DBs  

---

## 📊 Summary Table

| Component     | Library / Class           | Notes                                  |
|---------------|---------------------------|----------------------------------------|
| Loader        | `UnstructuredFileLoader`  | Parses PDFs → Document objects          |
| Splitter      | `CharacterTextSplitter`   | `chunk_size=1000`, `chunk_overlap=50` |
| Embeddings    | `HuggingFaceEmbeddings`   | Uses sentence-transformers             |
| Vector DB     | `Chroma`                  | Persistent DB at `vector_db/`          |
| Retriever     | `vectordb.as_retriever()` | KNN-based retrieval                    |
| LLM           | `ChatGroq`                | Model = `llama-3.3-70b-versatile`      |
| Orchestration | `RetrievalQA (LangChain)` | Returns answers + source documents     |

---

## 📚 Learning Insights
- ✂️ **Chunking matters**: balance between precision & recall (500–1500 tokens).  
- 🔗 **Embeddings**: choose domain-suited models for accuracy.  
- 💾 **Persist vectors**: saves time on re-runs.  
- ⚡ **Cost vs latency**: cloud vs local embeddings & LLM trade-offs.  
- 🔑 **Never commit secrets**: always use env vars.  
- 📑 **Return sources**: improves trustworthiness.  
- ✅ **Evaluate retrieval quality**: test retrieval with metrics + manual checks.  

---

## 🙌 Acknowledgements
- LangChain team & contributors 💡  
- Groq & `langchain-groq` ⚡  
- ChromaDB / `langchain-chroma` 📦  
- Hugging Face 🤗  
- `unstructured` project 📄  
- Example data: IPL 2025 public schedule 🏏  

---

## 🙏 Thank You
Thank you for exploring this project! 🚀  
Your feedback, ideas, and contributions are always welcome. 🌟  
Happy building with **RAG + LLMs** 🤖📚!  


