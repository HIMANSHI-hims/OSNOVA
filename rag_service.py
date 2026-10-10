"""
OSTutorLLM - RAG pipeline using OpenRouter Qwen + ChromaDB

Setup:
    python -m pip install chromadb sentence-transformers pypdf requests python-dotenv

CLI:
    python rag_service.py ingest
    python rag_service.py ask "What is a deadlock?"
    python rag_service.py plain "What is a deadlock?"
    python rag_service.py compare "Explain Round Robin"
    python rag_service.py mcq "process scheduling"
    python rag_service.py viva "deadlock"

Web:
    python app_server.py
"""

import glob
import json
import os
import sys
from typing import Dict, Generator, Iterable, List, Optional, Tuple

import chromadb
import requests
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------- Configuration ----------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

MODEL = os.environ.get(
    "OPENROUTER_MODEL",
    "qwen/qwen3-4b:free"
)
API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

CHUNK_SIZE = 800
OVERLAP = 150
TOP_K = 4

# Chat answers are direct. Set SHOW_SOURCES=true in .env to send the source
# list to the web interface again (e.g. for a "Show sources" toggle).
SHOW_SOURCES = os.environ.get("SHOW_SOURCES", "false").strip().lower() == "true"

# Turn off Qwen3's "thinking" so the user only sees the final answer.
DISABLE_THINKING = True

# NOTE: every line ends with a space so the joined string reads correctly.
SYSTEM_PROMPT = (
    "You are OSTutor, a friendly Operating Systems tutor. "
    "Answer the question directly, in a clear and simple way, like a teacher "
    "explaining to a student. Start with the answer immediately. "
    "Never say \"according to the context\", \"the notes say\", \"page\", "
    "\"source\" or \"document\", and never describe your reasoning. "
    "Use the material given to you as your own knowledge. "
    "If it does not contain the answer, reply only: "
    "\"This is not covered in the course material I have.\""
)

# ---------- Initialization ----------

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

embedder = SentenceTransformer(EMBED_MODEL)
db = chromadb.PersistentClient(path=DB_DIR)


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

    # Preserve the existing collection if there are no input files.
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
                with open(path, encoding="utf-8", errors="ignore") as file:
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
                metadata.append({
                    "source": name,
                    "page": page_no,
                })
                ids.append(f"{name}-p{page_no}-c{chunk_no}")

        if docs:
            vectors = embedder.encode(
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

        q_vector = embedder.encode([question]).tolist()

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
    """Format retrieved chunks for the model prompt.

    Only the chunk text is sent. File names and page numbers are NOT put in
    the prompt, otherwise the model repeats them ("page 2 of the notes...").
    """

    if not hits:
        return "No specific course notes retrieved."

    return "\n\n".join(document for document, _metadata in hits)


def build_sources(hits: List[Tuple[str, Dict]]) -> List[Dict]:
    """Build source information for the web interface.

    Returns an empty list unless SHOW_SOURCES=true, so the chat shows only
    the answer by default.
    """

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


# ---------- OpenRouter Streaming ----------

def strip_think(tokens: Iterable[str]) -> Generator[str, None, None]:
    """Remove <think>...</think> blocks from a stream of text pieces.

    Safety net: if the model still writes its reasoning inside the answer,
    it is dropped before reaching the user.
    """

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
                    # keep a possible partial closing tag, drop the rest
                    buffer = buffer[-(len(close_tag) - 1):]
                    break

                buffer = buffer[end + len(close_tag):]
                in_think = False

            else:
                start = buffer.find(open_tag)

                if start == -1:
                    # hold back a possible partial opening tag
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


def _raw_stream(
    system: str,
    user: str,
    model: Optional[str] = None,
) -> Generator[str, None, None]:
    """Stream raw generated text from the hosted Qwen model."""

    chosen_model = model or MODEL

    if not API_KEY:
        yield "Error: OPENROUTER_API_KEY is not configured."
        return

    # Qwen3 soft switch: asks the model not to think before answering.
    if DISABLE_THINKING and "qwen3" in chosen_model.lower():
        user = user.rstrip() + " /no_think"

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
        "max_tokens": 800,
        "stream": True,
    }

    if DISABLE_THINKING:
        # OpenRouter's switch for models that support reasoning.
        payload["reasoning"] = {"enabled": False}

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
                yield (
                    f"OpenRouter request failed "
                    f"(HTTP {response.status_code}): {details}"
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
                    yield f"OpenRouter API error: {event['error']}"
                    return

                choices = event.get("choices", [])

                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                token = delta.get("content")

                # Only "content" is used; the separate "reasoning" field
                # (if a provider sends one) is ignored on purpose.
                if isinstance(token, str) and token:
                    yield token

                elif isinstance(token, list):
                    for part in token:
                        if isinstance(part, dict):
                            text = part.get("text")
                            if text:
                                yield text

    except requests.RequestException as exc:
        yield f"OpenRouter connection error: {exc}"

    except Exception as exc:
        yield f"Unexpected model error: {exc}"


def stream_qwen(
    system: str,
    user: str,
    model: Optional[str] = None,
) -> Generator[str, None, None]:
    """Stream the final answer only (thinking removed, no leading blanks)."""

    started = False

    for piece in strip_think(_raw_stream(system, user, model)):
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
    """Print and return the generated answer."""

    full_answer = []

    for token in stream_qwen(system, user, model):
        print(token, end="", flush=True)
        full_answer.append(token)

    print()
    return "".join(full_answer)


# ---------- Status for Web App ----------

def get_available_models() -> List[str]:
    return [MODEL]


def get_status() -> Dict:
    """Return application status for the frontend."""

    files = []

    for extension in ("*.pdf", "*.txt", "*.md"):
        files.extend(
            os.path.basename(path)
            for path in glob.glob(
                os.path.join(DATA_DIR, extension)
            )
        )

    try:
        chunk_count = db.get_collection("os_notes").count()
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


# ---------- Web App Generation Functions ----------

def generate_rag(
    question: str,
    model: Optional[str] = None,
) -> Tuple[List[Dict], Generator[str, None, None]]:

    hits = retrieve(question)
    sources = build_sources(hits)

    prompt = (
        f"Context:\n{format_context(hits)}\n\n"
        f"Question: {question}"
    )

    return sources, stream_qwen(SYSTEM_PROMPT, prompt, model)


def generate_plain(
    question: str,
    model: Optional[str] = None,
) -> Tuple[List[Dict], Generator[str, None, None]]:

    system = (
        "You are a helpful Operating Systems tutor. "
        "Explain concepts clearly and concisely."
    )

    return [], stream_qwen(system, question, model)


def generate_mcq(
    topic: str,
    model: Optional[str] = None,
) -> Tuple[List[Dict], Generator[str, None, None]]:

    hits = retrieve(topic)
    sources = build_sources(hits)

    prompt = (
        f"Context:\n{format_context(hits)}\n\n"
        f"Create 5 multiple-choice questions on '{topic}' "
        "using ONLY the context. Give 4 options (A-D) for each, "
        "then list the correct answers with a one-line explanation "
        "at the end. Use clear Markdown headings."
    )

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

    # The command line still lists sources: handy for testing and screenshots.
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
    print("\n--- ANSWER (plain Qwen, no notes) ---")

    ask_qwen(
        "You are a helpful Operating Systems tutor.",
        question,
    )


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