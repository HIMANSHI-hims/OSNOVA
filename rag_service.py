"""
OSTutorLLM - unified RAG pipeline & service (Qwen3 via Ollama + ChromaDB)

Setup (once):
    pip install -U ollama chromadb sentence-transformers pypdf

CLI Usage:
    python rag_service.py ingest                      # read PDFs in DATA_DIR and build the vector DB
    python rag_service.py ask "What is a deadlock?"   # RAG answer + shows sources
    python rag_service.py plain "What is a deadlock?" # plain Qwen, no notes (for comparison)
    python rag_service.py compare "Explain Round Robin" # plain vs RAG, side by side
    python rag_service.py mcq "process scheduling"    # 5 MCQs from your notes
    python rag_service.py viva "deadlock"             # 5 viva questions from your notes

Web Usage:
    python app_server.py                              # runs web app at http://localhost:8000
"""

import glob
import os
import sys
from typing import Dict, Generator, List, Optional, Tuple

import chromadb
import ollama
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODEL = os.environ.get("OSTUTOR_MODEL", "qwen3:4b")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DATA_DIR = r"C:\Users\Himanshi\OneDrive\Desktop\OSTutor\data"
DB_DIR = "chroma_db"
CHUNK_SIZE = 800
OVERLAP = 150
TOP_K = 4

embedder = SentenceTransformer(EMBED_MODEL)
db = chromadb.PersistentClient(path=DB_DIR)

SYSTEM_PROMPT = (
    "You are OSTutor, a friendly Operating Systems tutor for B.Tech students. "
    "Explain simply, step by step, with a small example when useful. "
    "Answer ONLY using the context provided. If the context does not contain "
    "the answer, say: 'This is not covered in the course material I have.'"
)


def chunk_text(text: str) -> List[str]:
    """Split text into overlapping pieces so ideas are not cut in half."""
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start : start + CHUNK_SIZE].strip()
        if len(piece) > 50:
            chunks.append(piece)
        start += CHUNK_SIZE - OVERLAP
    return chunks


def ingest() -> Dict:
    """Read every PDF/TXT/MD in DATA_DIR, chunk it, embed the chunks, and store in ChromaDB."""
    try:
        db.delete_collection("os_notes")
    except Exception:
        pass
    collection = db.create_collection("os_notes")

    files = []
    for pattern in ("*.pdf", "*.txt", "*.md"):
        # Line that scans your DATA_DIR
        files += glob.glob(os.path.join(DATA_DIR, pattern))

    if not files:
        msg = f"No .pdf/.txt/.md files found in '{DATA_DIR}'. Add some and run again."
        print(msg)
        return {"success": False, "message": msg, "total_chunks": 0, "indexed_files": []}

    total = 0
    indexed_files = []
    for path in files:
        name = os.path.basename(path)
        pages = []
        try:
            if path.lower().endswith(".pdf"):
                reader = PdfReader(path)
                pages = [(n, p.extract_text() or "") for n, p in enumerate(reader.pages, start=1)]
            else:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    pages = [(1, f.read())]
        except Exception as e:
            print(f"Error reading {name}: {e}")
            continue

        docs, metas, ids = [], [], []
        for page_no, text in pages:
            for i, chunk in enumerate(chunk_text(text)):
                docs.append(chunk)
                metas.append({"source": name, "page": page_no})
                ids.append(f"{name}-p{page_no}-c{i}")

        if docs:
            vectors = embedder.encode(docs).tolist()
            collection.add(documents=docs, embeddings=vectors, metadatas=metas, ids=ids)
            total += len(docs)
            indexed_files.append({"filename": name, "chunks": len(docs)})
        print(f"Indexed {name}: {len(docs)} chunks")

    print(f"Done. {total} chunks stored.")
    return {
        "success": True,
        "message": f"Successfully indexed {len(indexed_files)} files ({total} chunks total).",
        "total_chunks": total,
        "indexed_files": indexed_files,
    }


ingest_notes = ingest


def retrieve(question: str, top_k: int = TOP_K) -> List[Tuple[str, Dict]]:
    """Find the chunks whose meaning is closest to the question."""
    try:
        collection = db.get_collection("os_notes")
        q_vec = embedder.encode([question]).tolist()
        result = collection.query(query_embeddings=q_vec, n_results=top_k)
        if result and result.get("documents") and result["documents"][0]:
            return list(zip(result["documents"][0], result["metadatas"][0]))
        return []
    except Exception as e:
        print(f"Retrieval note: {e}")
        return []


def format_context(hits: List[Tuple[str, Dict]]) -> str:
    if not hits:
        return "No specific course notes retrieved."
    return "\n\n".join(
        f"[Source: {m['source']}, page {m['page']}]\n{doc}" for doc, m in hits
    )


def ask_qwen(system: str, user: str, model: Optional[str] = None) -> str:
    """CLI stream printer."""
    chosen_model = model or MODEL
    stream = ollama.chat(
        model=chosen_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.3},
        think=False,
        stream=True,
    )
    full = ""
    for part in stream:
        token = part["message"]["content"]
        print(token, end="", flush=True)
        full += token
    print()
    return full


def stream_qwen(system: str, user: str, model: Optional[str] = None) -> Generator[str, None, None]:
    """Web SSE token generator."""
    chosen_model = model or MODEL
    try:
        stream = ollama.chat(
            model=chosen_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": 0.3},
            think=False,
            stream=True,
        )
        for part in stream:
            token = ""
            if isinstance(part, dict):
                token = part.get("message", {}).get("content", "")
            else:
                msg = getattr(part, "message", None)
                if msg:
                    token = getattr(msg, "content", "")
            if token:
                yield token
    except Exception as e:
        yield f"\n\n**[Error communicating with Ollama model '{chosen_model}']**: {str(e)}"


def get_available_models() -> List[str]:
    """Return available Ollama models."""
    try:
        res = ollama.list()
        models = []
        model_list = res.get("models", []) if isinstance(res, dict) else getattr(res, "models", [])
        for m in model_list:
            if isinstance(m, dict):
                models.append(m.get("model") or m.get("name", ""))
            else:
                models.append(getattr(m, "model", "") or getattr(m, "name", ""))
        return [m for m in models if m]
    except Exception:
        return [MODEL, "qwen:4b"]


def get_status() -> Dict:
    """Return system and note collection status for the web interface."""
    files = []
    for ext in ("*.pdf", "*.txt", "*.md"):
        files.extend([os.path.basename(f) for f in glob.glob(os.path.join(DATA_DIR, ext))])

    chunk_count = 0
    try:
        col = db.get_collection("os_notes")
        chunk_count = col.count()
    except Exception:
        chunk_count = 0

    return {
        "status": "ready",
        "model": MODEL,
        "available_models": get_available_models(),
        "data_files": files,
        "total_chunks": chunk_count,
        "data_dir": DATA_DIR,
    }


def generate_rag(question: str, model: Optional[str] = None) -> Tuple[List[Dict], Generator[str, None, None]]:
    hits = retrieve(question)
    sources = [
        {"source": m.get("source", "Unknown"), "page": m.get("page", 1), "snippet": doc[:160] + "..."}
        for doc, m in hits
    ]
    prompt = f"Context:\n{format_context(hits)}\n\nQuestion: {question}"
    return sources, stream_qwen(SYSTEM_PROMPT, prompt, model=model)


def generate_plain(question: str, model: Optional[str] = None) -> Tuple[List[Dict], Generator[str, None, None]]:
    system = "You are a helpful and knowledgeable Operating Systems tutor. Explain concepts clearly and concisely."
    return [], stream_qwen(system, question, model=model)


def generate_mcq(topic: str, model: Optional[str] = None) -> Tuple[List[Dict], Generator[str, None, None]]:
    hits = retrieve(topic)
    sources = [
        {"source": m.get("source", "Unknown"), "page": m.get("page", 1), "snippet": doc[:160] + "..."}
        for doc, m in hits
    ]
    prompt = (
        f"Context:\n{format_context(hits)}\n\n"
        f"Create 5 multiple-choice questions on '{topic}' using ONLY the context. "
        "Give 4 options (A-D) for each, then list the correct answers with a "
        "one-line explanation at the end. Use markdown formatting with clear headings."
    )
    return sources, stream_qwen(SYSTEM_PROMPT, prompt, model=model)


def generate_viva(topic: str, model: Optional[str] = None) -> Tuple[List[Dict], Generator[str, None, None]]:
    hits = retrieve(topic)
    sources = [
        {"source": m.get("source", "Unknown"), "page": m.get("page", 1), "snippet": doc[:160] + "..."}
        for doc, m in hits
    ]
    prompt = (
        f"Context:\n{format_context(hits)}\n\n"
        f"Create 5 viva questions on '{topic}' using ONLY the context, "
        "ordered easy to hard, each with a clear, concise model answer."
    )
    return sources, stream_qwen(SYSTEM_PROMPT, prompt, model=model)


def rag_answer(question: str):
    hits = retrieve(question)
    prompt = f"Context:\n{format_context(hits)}\n\nQuestion: {question}"
    print("\n--- ANSWER (with your notes) ---")
    ask_qwen(SYSTEM_PROMPT, prompt)
    print("\n--- SOURCES USED ---")
    for _, m in hits:
        print(f"  {m['source']} (page {m['page']})")


def plain_answer(question: str):
    print("\n--- ANSWER (plain Qwen, no notes) ---")
    ask_qwen("You are a helpful assistant.", question)


def cli_mcq(topic: str):
    hits = retrieve(topic)
    prompt = (
        f"Context:\n{format_context(hits)}\n\n"
        f"Create 5 multiple-choice questions on '{topic}' using ONLY the context. "
        "Give 4 options (A-D) for each, then list the correct answers with a "
        "one-line explanation at the end."
    )
    print(f"\n--- MCQs on {topic} ---")
    ask_qwen(SYSTEM_PROMPT, prompt)


def cli_viva(topic: str):
    hits = retrieve(topic)
    prompt = (
        f"Context:\n{format_context(hits)}\n\n"
        f"Create 5 viva questions on '{topic}' using ONLY the context, "
        "ordered easy to hard, each with a short model answer."
    )
    print(f"\n--- Viva questions on {topic} ---")
    ask_qwen(SYSTEM_PROMPT, prompt)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit()

    command = sys.argv[1]
    arg = " ".join(sys.argv[2:])

    if command == "ingest":
        ingest()
    elif command == "ask":
        rag_answer(arg)
    elif command == "plain":
        plain_answer(arg)
    elif command == "compare":
        plain_answer(arg)
        rag_answer(arg)
    elif command == "mcq":
        cli_mcq(arg)
    elif command == "viva":
        cli_viva(arg)
    else:
        print(__doc__)
