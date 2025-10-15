# 🤖 AI Data Assistant

An intelligent document-aware chatbot built with **Streamlit**, **LangChain**, **FAISS**, and **OpenAI GPT models**.  
Upload your PDFs, Word, Text, or PowerPoint files — and chat with your data in natural language.

---

## 📘 Technical Documentation

A detailed step-by-step explanation of the application’s design, logic, and architecture can be found in the file:  
📄 **AI Data Assistant.docx**

---


## 🌟 Features

- 📂 **Multi-format document support** — PDF, DOCX, TXT, PPTX  
- 🧠 **Context-aware Q&A** using GPT-4 and LangChain RetrievalQA  
- 🔍 **FAISS-based vector search** for fast and semantic retrieval  
- 🧾 **Automatic summarization** for large documents  
- 💬 **Interactive Streamlit chat interface**  
- 🖼️ **OCR fallback** for scanned PDFs (via Tesseract)  
- ⚙️ **Session memory** for conversational context  

---

## 🏗️ Architecture
```
User → Upload Files → Text Extraction → Chunking & Embedding → FAISS Indexing
↓
Chat Query → Context Retrieval → GPT-4 Response Generation → Display
```
---


## 🧰 Tech Stack

| Layer | Technology |
|-------|-------------|
| **Frontend** | Streamlit |
| **LLM Orchestration** | LangChain |
| **Vector Store** | FAISS |
| **Embedding Model** | OpenAI Embeddings |
| **Language Model** | GPT-3.5 / GPT-4 |
| **Document Parsing** | pdfplumber, python-docx, python-pptx |
| **OCR** | pytesseract + Pillow |
| **Environment Management** | python-dotenv |

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/ai-data-assistant.git
cd ai-data-assistant

2️⃣ Create and activate a virtual environment
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add your OpenAI API key
Create a .env file in the project root:
OPENAI_API_KEY=your_openai_api_key_here

5️⃣ Run the Streamlit app
streamlit run DataAssistantChatbot.py

Then open the provided local URL in your browser.
```
---

## 🧠 How It Works

1. **Text Extraction** — Uses format-specific libraries and OCR fallback.  
2. **Chunking** — Splits large text into 1000-character overlapping chunks.  
3. **Embedding** — Creates semantic embeddings using OpenAI models.  
4. **Indexing** — Stores embeddings in FAISS for efficient retrieval.  
5. **Query Processing** — Fetches relevant chunks and generates a GPT-based response.

---

## 🚀 Future Improvements

- Persistent vector databases (e.g., Pinecone, Chroma)  
- Metadata-based search and filtering  
- Multi-user session management  
- Enhanced image and table parsing  
- Streamlit dashboard for analytics  

---
