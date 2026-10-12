# OSNOVA 🚀
### AI-Powered Operating Systems Tutor

**Understand the system. Master the concepts.**

OSNOVA is an AI-powered Operating Systems tutor that combines Retrieval-Augmented Generation (RAG) with a large language model to help students understand complex OS concepts, explore their study notes, and prepare for examinations.

Instead of relying solely on general-purpose AI responses, OSNOVA retrieves relevant information from your own study materials and uses it to generate contextual, easy-to-understand answers.

## ✨ Features

- **📚 Context-Aware Learning:** Retrieves relevant information from uploaded study materials using semantic search.
- **🤖 AI-Powered Answers:** Uses the Gemini API to explain Operating Systems concepts with clear, student-friendly responses.
- **⚡ Streaming Responses:** Displays generated answers progressively instead of waiting for the entire response.
- **📝 MCQ Generation:** Generates multiple-choice questions for practice and revision.
- **🎓 Viva Preparation:** Creates viva questions with concise model answers.
- **🔍 Semantic Retrieval:** Uses Sentence Transformers and ChromaDB to find relevant content from indexed notes.
- **📄 Multiple Document Formats:** Supports PDF, TXT, and Markdown study materials.
- **💻 Interactive Web Interface:** Provides a dedicated frontend for interacting with the tutor.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Large Language Model | Google Gemini API |
| RAG Pipeline | Custom Python implementation |
| Embeddings | Sentence Transformers — `BAAI/bge-small-en-v1.5` |
| Vector Database | ChromaDB |
| Document Processing | pypdf |
| Backend Server | Python HTTP server |
| Frontend | HTML, CSS, JavaScript |
| Response Streaming | Server-Sent Events (SSE) |

## 🧠 How OSNOVA Works

OSNOVA follows a Retrieval-Augmented Generation workflow.

1. **Document Ingestion:** Study materials are loaded from PDF, TXT, and Markdown files.
2. **Text Chunking:** Documents are divided into smaller, overlapping chunks.
3. **Embedding Generation:** Sentence Transformers converts document chunks into vector embeddings.
4. **Vector Storage:** ChromaDB stores the embeddings alongside their associated text and metadata.
5. **Semantic Retrieval:** When a student asks a question, OSNOVA retrieves the most relevant chunks.
6. **Context Construction:** The retrieved content is included in the prompt sent to Gemini.
7. **Response Generation:** Gemini generates an answer using the retrieved context and its general knowledge where appropriate.
8. **Streaming:** The generated response is delivered progressively to the frontend.

For questions that do not require retrieval, OSNOVA also supports a plain AI response mode.

## 🏗️ Architecture

```text
             Student
                |
                v
        OSNOVA Web Interface
          HTML / CSS / JS
                |
                v
        Python Backend Server
                |
                v
         Request Mode Router
          /             \
         v               v
   RAG Pipeline       Plain AI Mode
         |
         v
   Query Embedding
         |
         v
      ChromaDB
         |
         v
  Relevant Study Chunks
         |
         v
   Prompt Construction
         |
         v
      Gemini API
         |
         v
   Streaming Response
         |
         v
      Student
```

## 📂 Project Structure

```text
OSNOVA/
│
├── data/                 # Source study materials
├── chroma_db/            # Persistent vector database
├── app_server.py         # Web server and API endpoints
├── rag_service.py        # RAG, ingestion and LLM integration
├── index.html            # Web interface
├── app.js                # Frontend application logic
├── style.css             # Frontend styling
├── .env                  # API configuration (not committed)
├── .gitignore
└── README.md
```

*The exact frontend filenames may differ depending on your current project structure.*

## ⚙️ Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version
- Git
- A Google Gemini API key
- An internet connection for API access and initial model downloads

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/OSNOVA.git
cd OSNOVA
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install chromadb sentence-transformers pypdf requests python-dotenv
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_supported_gemini_model
SHOW_SOURCES=false
```

Replace the placeholder values with your actual configuration. Use a Gemini model identifier supported by the API.

**Security:** Never commit your `.env` file or expose your API key in frontend JavaScript or public repositories.

### 5. Add study materials

Place your OS study materials in the `data/` directory.

Supported formats:

- PDF (`.pdf`)
- Text (`.txt`)
- Markdown (`.md`)

### 6. Index your study materials

```bash
python rag_service.py ingest
```

This processes the documents, generates embeddings, and stores them in ChromaDB.

Run ingestion again whenever you add or modify your study materials.

### 7. Start OSNOVA

```bash
python app_server.py
```

Open the following address in your browser:

**http://127.0.0.1:8000**

## 💬 Available Modes

| Mode | Purpose |
|---|---|
| RAG | Answers questions using relevant indexed study materials and general knowledge when appropriate |
| Plain AI | Answers questions without retrieving study notes |
| MCQ | Generates practice multiple-choice questions |
| Viva | Generates viva questions with model answers |

## 🖥️ CLI Usage

OSNOVA also provides command-line access through `rag_service.py`.

Ask a question using your indexed notes:

```bash
python rag_service.py ask "What is a deadlock?"
```

Ask a general question:

```bash
python rag_service.py plain "Explain process scheduling."
```

Generate practice questions:

```bash
python rag_service.py mcq "Memory Management"
```

Prepare for a viva:

```bash
python rag_service.py viva "Process Synchronization"
```

Compare general AI answers with RAG-based answers:

```bash
python rag_service.py compare "Explain Round Robin Scheduling"
```

## 🔮 Future Improvements

- Conversation memory and multi-turn contextual learning
- Adjustable question difficulty and exam-oriented practice
- More robust source citations and page-level references
- Improved response-length controls and generation diagnostics
- Support for additional subjects and document collections
- Automated evaluation of retrieval relevance and answer quality
- Deployment with secure environment-variable management

## 🎯 Project Objective

OSNOVA explores how Retrieval-Augmented Generation can make AI-assisted learning more grounded in a student's own study materials while preserving the flexibility of a general-purpose language model.

The project brings together semantic search, vector databases, document processing, API integration, and streaming web interfaces in a single educational application.

## 👨‍💻 Author

**YOUR NAME**

- GitHub: [YOUR-USERNAME](https://github.com/YOUR-USERNAME)
- LinkedIn: [Your LinkedIn Profile](YOUR-LINKEDIN-URL)

---

*OSNOVA — Learn smarter. Understand deeper.*
