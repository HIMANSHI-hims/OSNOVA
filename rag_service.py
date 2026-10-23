"""
OSNOVA - Qwen3:4B for note-grounded study + Gemini for plain chat

Web:
    python app_server.py

CLI:
    python rag_service.py ingest
    python rag_service.py ask "What is a deadlock?"
    python rag_service.py plain "What is a deadlock?"
    python rag_service.py compare "Explain Round Robin"
    python rag_service.py mcq "process scheduling"
    python rag_service.py viva "deadlock"
"""

import glob
import json
import os
import sys
import time
from typing import Dict, Generator, Iterable, List, Optional, Tuple

import chromadb
import requests
from dotenv import load_dotenv
from pypdf import PdfReader


# ---------- Configuration ----------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Qwen3:4B via Ollama handles note-grounded study modes.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")

# Gemini configuration is retained as an optional helper; the app defaults to Qwen.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
API_KEY = os.environ.get("GEMINI_API_KEY")

API_URL = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/openai/chat/completions"
)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

CHUNK_SIZE = 800
OVERLAP = 150
TOP_K = 4

SHOW_SOURCES = (
    os.environ.get("SHOW_SOURCES", "false").strip().lower() == "true"
)

SYSTEM_PROMPT = """
You are OSNOVA, a direct, exam-focused Operating Systems tutor.

GIVE THE SOURCES BEFORE DISPLAYING THE ANSWER.
"""

# ---------- Initialization ----------

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

db = chromadb.PersistentClient(path=DB_DIR)

_embedder = None


def get_embedder():
    """Load and reuse the embedding model."""
    global _embedder

    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        print(f"[OSTutor] Loading embedding model: {EMBED_MODEL}")
        start = time.perf_counter()

        _embedder = SentenceTransformer(EMBED_MODEL)

        print(
            f"[OSTutor] Embedding model loaded in "
            f"{time.perf_counter() - start:.2f}s"
        )

    return _embedder


def initialize():
    """Warm up the embedding model before accepting requests."""
    print("[OSTutor] Initializing RAG components...")
    start = time.perf_counter()

    embedder = get_embedder()

    embedder.encode(
        ["Operating Systems initialization"],
        show_progress_bar=False,
    )

    print("[OSTutor] Embedding model warmed up.")

    try:
        collection = db.get_collection("os_notes")
        print(
            f"[OSTutor] ChromaDB ready: "
            f"{collection.count()} chunks available."
        )
    except Exception:
        print(
            "[OSTutor] ChromaDB collection 'os_notes' "
            "not found. Run ingestion if needed."
        )

    print(
        f"[OSTutor] Initialization completed in "
        f"{time.perf_counter() - start:.2f}s"
    )


# ---------- Text Processing ----------

def chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    step = CHUNK_SIZE - OVERLAP

    while start < len(text):
        piece = text[start:start + CHUNK_SIZE].strip()

        if len(piece) > 50:
            chunks.append(piece)

        start += step

    return chunks


# ---------- Ingestion ----------

def ingest() -> Dict:
    """Read PDF/TXT/MD files and rebuild the ChromaDB collection."""
    files = []

    for pattern in ("*.pdf", "*.txt", "*.md"):
        files.extend(glob.glob(os.path.join(DATA_DIR, pattern)))

    if not files:
        msg = (
            f"No .pdf/.txt/.md files found in '{DATA_DIR}'. "
            "Add course notes and run ingestion again."
        )
        print(msg)

        return {
            "success": False,
            "message": msg,
            "total_chunks": 0,
            "indexed_files": [],
        }

    try:
        db.delete_collection("os_notes")
    except Exception:
        pass

    collection = db.create_collection("os_notes")

    total = 0
    indexed_files = []

    for path in files:
        name = os.path.basename(path)

        try:
            if path.lower().endswith(".pdf"):
                reader = PdfReader(path)
                pages = [
                    (page_no, page.extract_text() or "")
                    for page_no, page in enumerate(reader.pages, start=1)
                ]
            else:
                with open(
                    path, encoding="utf-8", errors="ignore"
                ) as file:
                    pages = [(1, file.read())]

        except Exception as exc:
            print(f"Error reading {name}: {exc}")
            continue

        docs = []
        metadata = []
        ids = []

        for page_no, page_text in pages:
            for chunk_no, chunk in enumerate(chunk_text(page_text)):
                docs.append(chunk)
                metadata.append({"source": name, "page": page_no})
                ids.append(f"{name}-p{page_no}-c{chunk_no}")

        if docs:
            vectors = get_embedder().encode(
                docs,
                show_progress_bar=False,
            ).tolist()

            collection.add(
                documents=docs,
                embeddings=vectors,
                metadatas=metadata,
                ids=ids,
            )

            total += len(docs)
            indexed_files.append({
                "filename": name,
                "chunks": len(docs),
            })

        print(f"Indexed {name}: {len(docs)} chunks")

    message = (
        f"Successfully indexed {len(indexed_files)} files "
        f"({total} chunks total)."
    )

    print(message)

    return {
        "success": True,
        "message": message,
        "total_chunks": total,
        "indexed_files": indexed_files,
    }


ingest_notes = ingest


# ---------- Retrieval ----------

def retrieve(
    question: str,
    top_k: int = TOP_K,
) -> List[Tuple[str, Dict]]:
    """Retrieve relevant chunks from the notes."""
    try:
        collection = db.get_collection("os_notes")
        count = collection.count()

        if count == 0:
            return []

        q_vector = get_embedder().encode([question]).tolist()

        result = collection.query(
            query_embeddings=q_vector,
            n_results=min(top_k, count),
        )

        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]

        if documents[0]:
            return list(zip(documents[0], metadatas[0]))

        return []

    except Exception as exc:
        print(f"Retrieval error: {exc}")
        return []


def format_context(hits: List[Tuple[str, Dict]]) -> str:
    """Format retrieved chunks for the model prompt."""
    if not hits:
        return "No specific course notes retrieved."

    return "\n\n".join(document for document, _metadata in hits)


def build_sources(hits: List[Tuple[str, Dict]]) -> List[Dict]:
    """Build optional source information for the web interface."""
    if not SHOW_SOURCES:
        return []

    return [
        {
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page", 1),
            "snippet": document[:160] + (
                "..." if len(document) > 160 else ""
            ),
        }
        for document, metadata in hits
    ]


# ---------- Model Streaming ----------

def strip_think(
    tokens: Iterable[str],
) -> Generator[str, None, None]:
    """Remove explicit <think>...</think> blocks from generated text."""
    open_tag, close_tag = "<think>", "</think>"
    buffer = ""
    in_think = False

    for token in tokens:
        buffer += token
        out = ""

        while True:
            if in_think:
                end = buffer.find(close_tag)
                if end == -1:
                    buffer = buffer[-(len(close_tag) - 1):]
                    break
                buffer = buffer[end + len(close_tag):]
                in_think = False
            else:
                start = buffer.find(open_tag)
                if start == -1:
                    keep = len(open_tag) - 1
                    if len(buffer) > keep:
                        out += buffer[:-keep]
                        buffer = buffer[-keep:]
                    break
                out += buffer[:start]
                buffer = buffer[start + len(open_tag):]
                in_think = True

        if out:
            yield out

    if buffer and not in_think:
        yield buffer


def _ollama_stream(
    system: str,
    user: str,
    model: Optional[str] = None,
) -> Generator[str, None, None]:
    """Stream concise local Qwen output from Ollama."""

    chosen_model = OLLAMA_MODEL
    user_lower = user.lower()

    if "multiple-choice questions" in user_lower:
        max_tokens = 3800
    elif "viva questions" in user_lower:
        max_tokens = 2500
    else:
        max_tokens = 2000

    payload = {
        "model": chosen_model,
        "messages": [
    {
        "role": "system",
        "content": (
            system
            + "\n\nRESPONSE RULES:\n"
            "Answer the user's actual question directly.\n"
            "Start with the answer, not your approach or thought process.\n"
            "Do not narrate reasoning, planning, or what you are going to do.\n"
            "Use the supplied context when relevant.\n"
            "If the context does not contain the answer, say so clearly.\n"
            "Keep the answer concise, clear, and suitable for a student."
        ),
    },
    {"role": "user", "content": user},
],
        "stream": True,
        "think": False,
        "options": {
    "temperature": 0.1,
    "num_predict": max_tokens,
    "repeat_penalty": 1.1,
},
    }

    try:
        with requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=(10, 300),
        ) as response:

            if not response.ok:
                details = response.text[:800]
                print(
                    f"Ollama request failed "
                    f"(HTTP {response.status_code}): {details}"
                )
                yield (
                    f"Local Qwen request failed "
                    f"(HTTP {response.status_code}). "
                    "Check that Ollama is running and the model is installed."
                )
                return

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                if isinstance(line, bytes):
                    line = line.decode("utf-8")

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event.get("error"):
                    print(f"Ollama error: {event['error']}")
                    yield f"Local Qwen error: {event['error']}"
                    return

                message = event.get("message") or {}
                token = message.get("content")

                if isinstance(token, str) and token:
                    yield token

                if event.get("done"):
                    break

    except requests.ConnectionError as exc:
        print(f"Ollama connection error: {exc}")
        yield (
            "Cannot connect to Ollama. Start Ollama and make sure "
            f"the model '{OLLAMA_MODEL}' is installed."
        )

    except requests.RequestException as exc:
        print(f"Ollama request error: {exc}")
        yield (
            "Local Qwen request failed. "
            "Check the Ollama service and backend logs."
        )

    except Exception as exc:
        print(f"Unexpected Ollama error: {exc}")
        yield "Unexpected local Qwen error. Check the backend logs."



def _gemini_stream(
    system: str,
    user: str,
    model: Optional[str] = None,
) -> Generator[str, None, None]:
    """Stream general-chat output from the existing Gemini API."""
    # The frontend may still send its selected Qwen model for every mode.
    # Never pass that value to Gemini; only accept an explicit Gemini model.
    chosen_model = (
        model if isinstance(model, str) and model.startswith("gemini")
        else GEMINI_MODEL
    )

    if not API_KEY:
        yield "Error: GEMINI_API_KEY is not configured."
        return

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
        "stream": True,
    }

    try:
        with requests.post(
            API_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=(15, 120),
        ) as response:
            if not response.ok:
                details = response.text[:1000]
                print(f"Gemini request failed (HTTP {response.status_code}): {details}")
                yield (
                    f"Gemini request failed (HTTP {response.status_code}). "
                    "Check the backend logs for details."
                )
                return

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if event.get("error"):
                    print(f"Gemini API error: {event['error']}")
                    yield "Gemini API returned an error. Check backend logs."
                    return
                choices = event.get("choices", [])
                if not choices:
                    continue
                token = (choices[0].get("delta") or {}).get("content")
                if isinstance(token, str) and token:
                    yield token
                elif isinstance(token, list):
                    for part in token:
                        if isinstance(part, dict) and part.get("text"):
                            yield part["text"]

    except requests.RequestException as exc:
        print(f"Gemini connection error: {exc}")
        yield "Gemini connection error. Check your network and API endpoint."
    except Exception as exc:
        print(f"Unexpected Gemini error: {exc}")
        yield "Unexpected Gemini error. Check the backend logs."


def stream_qwen(
    system: str,
    user: str,
    model: Optional[str] = None,
) -> Generator[str, None, None]:
    """Compatibility wrapper: study/RAG features use local Qwen via Ollama."""
    started = False
    for piece in strip_think(_ollama_stream(system, user, model)):
        if not started:
            piece = piece.lstrip()
            if not piece:
                continue
            started = True
        yield piece


def stream_gemini(
    system: str,
    user: str,
    model: Optional[str] = None,
) -> Generator[str, None, None]:
    """Stream general AI chat through Gemini, keeping its API unchanged."""
    started = False
    for piece in strip_think(_gemini_stream(system, user, model)):
        if not started:
            piece = piece.lstrip()
            if not piece:
                continue
            started = True
        yield piece


def ask_qwen(
    system: str,
    user: str,
    model: Optional[str] = None,
) -> str:
    """Print and return a local Qwen answer (kept for CLI compatibility)."""
    full_answer = []
    for token in stream_qwen(system, user, model):
        print(token, end="", flush=True)
        full_answer.append(token)
    print()
    return "".join(full_answer)


def ask_gemini(
    system: str,
    user: str,
    model: Optional[str] = None,
) -> str:
    """Print and return a Gemini answer."""
    full_answer = []
    for token in stream_gemini(system, user, model):
        print(token, end="", flush=True)
        full_answer.append(token)
    print()
    return "".join(full_answer)


# ---------- Status for Web App ----------

def get_available_models() -> List[str]:
    """List the configured local and API models used by OSNOVA."""
    return [OLLAMA_MODEL, GEMINI_MODEL]


def get_status() -> Dict:
    """Return application status for the frontend."""
    files = []

    for extension in ("*.pdf", "*.txt", "*.md"):
        files.extend(
            os.path.basename(path)
            for path in glob.glob(os.path.join(DATA_DIR, extension))
        )

    try:
        chunk_count = db.get_collection("os_notes").count()
    except Exception:
        chunk_count = 0

    return {
        "status": "ready",
        "model": f"{OLLAMA_MODEL} (RAG/MCQ/Viva) + {GEMINI_MODEL} (Plain Chat)",
        "available_models": get_available_models(),
        "primary_model": OLLAMA_MODEL,
        "mode_models": {
            "rag": OLLAMA_MODEL,
            "mcq": OLLAMA_MODEL,
            "viva": OLLAMA_MODEL,
            "plain": GEMINI_MODEL,
            "compare": f"{OLLAMA_MODEL} + {GEMINI_MODEL}",
        },
        "data_files": files,
        "total_chunks": chunk_count,
        "data_dir": DATA_DIR,
    }


# ---------- Web App Generation Functions ----------

def generate_rag(
    question: str,
    model: Optional[str] = None,
) -> Tuple[List[Dict], Generator[str, None, None]]:
    hits = retrieve(question)
    sources = build_sources(hits)
    prompt = (
    "Answer the question using the context below.\n"
    "or repeat the question or context.\n\n"
    f"<context>\n{format_context(hits)}\n</context>\n\n"
    f"<question>\n{question}\n</question>\n\n"
    "Answer:"
)

    return sources, stream_qwen(SYSTEM_PROMPT, prompt, model)


def generate_plain(
    question: str,
    model: Optional[str] = None,
) -> Tuple[List[Dict], Generator[str, None, None]]:
    """Answer general questions with Gemini, without retrieving notes."""

    system = """
You are OSNOVA, a direct, exam-focused tutor.

Return ONLY the final answer.
Start immediately with the answer itself.
Never show reasoning, planning, analysis, or internal monologue.
Never describe the question or explain what you are doing.
Never use phrases like "Let's craft the answer",
"The user is asking", "From the context", or "We need to answer".
No preamble, casual talk, filler, or closing remarks.
For definitions, use 1-2 sentences.
For simple factual questions, use at most 3 sentences.
Use simple, clear, exam-friendly language.
For code requests, provide the code directly.
"""

    return [], stream_gemini(system, question, model)


def generate_mcq(
    topic: str,
    model: Optional[str] = None,
) -> Tuple[List[Dict], Generator[str, None, None]]:
    hits = retrieve(topic)
    sources = build_sources(hits)

    prompt = f"Question: {question}"

    return sources, stream_qwen(SYSTEM_PROMPT, prompt, model)


def generate_viva(
    topic: str,
    model: Optional[str] = None,
) -> Tuple[List[Dict], Generator[str, None, None]]:
    hits = retrieve(topic)
    sources = build_sources(hits)

    prompt = (
        f"Context:\n{format_context(hits)}\n\n"
        f"Create 5 viva questions on '{topic}' using ONLY "
        "the context, ordered easy to hard, each with a "
        "clear, concise model answer."
    )

    return sources, stream_qwen(SYSTEM_PROMPT, prompt, model)


# ---------- CLI Commands ----------

def rag_answer(question: str):
    hits = retrieve(question)
    prompt = (
        f"Context:\n{format_context(hits)}\n\n"
        f"Question: {question}"
    )

    print("\n--- ANSWER (with your notes) ---")
    ask_qwen(SYSTEM_PROMPT, prompt)

    print("\n--- SOURCES USED ---")

    if not hits:
        print("No relevant sources retrieved.")
    else:
        for _, metadata in hits:
            print(
                f"  {metadata.get('source', 'Unknown')} "
                f"(page {metadata.get('page', 1)})"
            )


def plain_answer(question: str):
    print("\\n--- ANSWER (local Qwen, no notes) ---")
    ask_qwen(SYSTEM_PROMPT, question)


def cli_mcq(topic: str):
    hits = retrieve(topic)

    prompt = (
        f"Context:\n{format_context(hits)}\n\n"
        f"Create 5 multiple-choice questions on '{topic}' "
        "using ONLY the context. Give 4 options (A-D) for each, "
        "then list the correct answers with a one-line explanation."
    )

    print(f"\n--- MCQs on {topic} ---")
    ask_qwen(SYSTEM_PROMPT, prompt)


def cli_viva(topic: str):
    hits = retrieve(topic)

    prompt = (
        f"Context:\n{format_context(hits)}\n\n"
        f"Create 5 viva questions on '{topic}' using ONLY "
        "the context, ordered easy to hard, each with a short answer."
    )

    print(f"\n--- Viva questions on {topic} ---")
    ask_qwen(SYSTEM_PROMPT, prompt)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]
    argument = " ".join(sys.argv[2:])

    if command == "ingest":
        ingest()
    elif command == "ask":
        rag_answer(argument)
    elif command == "plain":
        plain_answer(argument)
    elif command == "compare":
        plain_answer(argument)
        rag_answer(argument)
    elif command == "mcq":
        cli_mcq(argument)
    elif command == "viva":
        cli_viva(argument)
    else:
        print(__doc__)