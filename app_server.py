"""
app_server.py - OSNOVA frontend and local Qwen backend.
Serves the web UI and provides streaming SSE endpoints connected to rag_service.py.
"""

import argparse
import json
import os
import sys
import time
import traceback
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import rag_service


class OSTutorHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=CURRENT_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, format_string, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {self.client_address[0]} - {format_string % args}")

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            self.handle_api_status()
        elif path == "/api/models":
            self.handle_api_models()
        elif path in ("/", ""):
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/chat/stream":
            self.handle_chat_stream()
        elif path == "/api/ingest":
            self.handle_api_ingest()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def _read_json_body(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            return json.loads(self.rfile.read(content_length).decode("utf-8"))
        return {}

    def _send_json(self, data, status=HTTPStatus.OK):
        response = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def handle_api_status(self):
        try:
            self._send_json(rag_service.get_status())
        except Exception as exc:
            print(f"[OSNOVA] Status error: {exc}")
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_api_models(self):
        try:
            self._send_json({"models": rag_service.get_available_models()})
        except Exception as exc:
            print(f"[OSNOVA] Models error: {exc}")
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_api_ingest(self):
        try:
            start = time.perf_counter()
            print("[OSNOVA] Starting note ingestion...")
            result = rag_service.ingest_notes()
            print(f"[OSNOVA] Ingestion completed in {time.perf_counter() - start:.2f}s")
            self._send_json(result)
        except Exception as exc:
            print(f"[OSNOVA] Ingestion error: {exc}")
            traceback.print_exc()
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_chat_stream(self):
        start = time.perf_counter()
        print("\n[OSNOVA] New chat request received")
        headers_sent = False
        try:
            body = self._read_json_body()
            prompt = body.get("prompt", "").strip()
            mode = body.get("mode", "rag")
            requested_model = body.get("model") or None

            if not prompt:
                self._send_json({"error": "Missing prompt in request body"}, HTTPStatus.BAD_REQUEST)
                return

            # Keep both models: Qwen handles note-grounded study modes;
            # Gemini handles Plain Chat. Model routing is logged honestly.
            print(f"[OSNOVA] Mode: {mode}")
            if mode in ("rag", "mcq", "viva"):
                actual_model = rag_service.OLLAMA_MODEL
            elif mode == "plain":
                actual_model = rag_service.GEMINI_MODEL
            elif mode == "compare":
                actual_model = f"{rag_service.OLLAMA_MODEL} + {rag_service.GEMINI_MODEL}"
            else:
                actual_model = rag_service.OLLAMA_MODEL
            print(f"[OSNOVA] Model route: {actual_model}")

            setup_start = time.perf_counter()
            if mode == "rag":
                sources, token_stream = rag_service.generate_rag(prompt)
            elif mode == "plain":
                sources, token_stream = rag_service.generate_plain(prompt)
            elif mode == "mcq":
                sources, token_stream = rag_service.generate_mcq(prompt)
            elif mode == "viva":
                sources, token_stream = rag_service.generate_viva(prompt)
            elif mode == "compare":
                rag_sources, rag_tokens = rag_service.generate_rag(prompt)
                _plain_sources, plain_tokens = rag_service.generate_plain(prompt)

                def combined_compare():
                    yield f"## Answer using your OS notes ({rag_service.OLLAMA_MODEL})\n\n"
                    yield from rag_tokens
                    yield f"\n\n---\n\n## General answer ({rag_service.GEMINI_MODEL})\n\n"
                    yield from plain_tokens

                sources, token_stream = rag_sources, combined_compare()
            else:
                print(f"[OSNOVA] Unknown mode '{mode}', using RAG mode.")
                sources, token_stream = rag_service.generate_rag(prompt)

            print(f"[OSNOVA] Generation setup took {time.perf_counter() - setup_start:.2f}s")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            headers_sent = True

            init_event = json.dumps({"type": "sources", "sources": sources})
            self.wfile.write(f"data: {init_event}\n\n".encode("utf-8"))
            self.wfile.flush()

            first_token = True
            chunk_count = 0
            for token in token_stream:
                if first_token:
                    print(f"[OSNOVA] First output arrived after {time.perf_counter() - start:.2f}s")
                    first_token = False
                event = json.dumps({"type": "token", "token": token})
                self.wfile.write(f"data: {event}\n\n".encode("utf-8"))
                self.wfile.flush()
                chunk_count += 1

            done_event = json.dumps({"type": "done"})
            self.wfile.write(f"data: {done_event}\n\n".encode("utf-8"))
            self.wfile.flush()
            print(f"[OSNOVA] Completed in {time.perf_counter() - start:.2f}s ({chunk_count} chunks)")

        except (BrokenPipeError, ConnectionResetError):
            print("[OSNOVA] Client disconnected during streaming")
        except Exception as exc:
            print(f"[OSNOVA] Chat error: {exc}")
            traceback.print_exc()
            if headers_sent:
                try:
                    event = json.dumps({"type": "error", "error": str(exc)})
                    self.wfile.write(f"data: {event}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            else:
                try:
                    self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass


def run_server(port=8000):
    httpd = ThreadingHTTPServer(("0.0.0.0", port), OSTutorHandler)
    print(f"OSNOVA server running on port {port}")
    print(f"Open http://127.0.0.1:{port} in your browser")
    print(f"Primary model: {rag_service.OLLAMA_MODEL} via local Ollama (all modes)")
    print("Press Ctrl+C to stop the server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSNOVA Web Server")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on")
    args = parser.parse_args()
    run_server(args.port or int(os.environ.get("PORT", 8000)))
