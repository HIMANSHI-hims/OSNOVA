"""
app_server.py - Zero-dependency Python server for OSTutor Frontend & RAG Backend
Serves the web UI and provides streaming SSE endpoints connected to rag_service.py.
Usage:
    python app_server.py [--port 8000]
"""

import argparse
import json
import mimetypes
import os
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# Ensure current directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import rag_service


class OSTutorHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=CURRENT_DIR, **kwargs)

    def end_headers(self):
        # Enable CORS for local development
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self.handle_api_status()
        elif path == "/api/models":
            self.handle_api_models()
        elif path == "/" or path == "":
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/chat/stream":
            self.handle_chat_stream()
        elif path == "/api/ingest":
            self.handle_api_ingest()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def _read_json_body(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            body = self.rfile.read(content_length).decode("utf-8")
            return json.loads(body)
        return {}

    def handle_api_status(self):
        try:
            status = rag_service.get_status()
            data = json.dumps(status).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def handle_api_models(self):
        try:
            models = rag_service.get_available_models()
            data = json.dumps({"models": models}).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def handle_api_ingest(self):
        try:
            result = rag_service.ingest_notes()
            data = json.dumps(result).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def handle_chat_stream(self):
        """Stream SSE tokens to the web client."""
        try:
            body = self._read_json_body()
            prompt = body.get("prompt", "").strip()
            mode = body.get("mode", "rag")  # 'rag', 'plain', 'mcq', 'viva'
            model = body.get("model") or None

            if not prompt:
                self.send_error(HTTPStatus.BAD_REQUEST, "Missing prompt in request body")
                return

            # Choose generator based on mode
            if mode == "rag":
                sources, token_stream = rag_service.generate_rag(prompt, model=model)
            elif mode == "plain":
                sources, token_stream = rag_service.generate_plain(prompt, model=model)
            elif mode == "mcq":
                sources, token_stream = rag_service.generate_mcq(prompt, model=model)
            elif mode == "viva":
                sources, token_stream = rag_service.generate_viva(prompt, model=model)
            else:
                sources, token_stream = rag_service.generate_rag(prompt, model=model)

            # Start SSE Response
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            # First event: Send sources list
            init_event = json.dumps({"type": "sources", "sources": sources})
            self.wfile.write(f"data: {init_event}\n\n".encode("utf-8"))
            self.wfile.flush()

            # Token events: Stream tokens as generated
            for token in token_stream:
                token_event = json.dumps({"type": "token", "token": token})
                self.wfile.write(f"data: {token_event}\n\n".encode("utf-8"))
                self.wfile.flush()

            # Final event: Done
            done_event = json.dumps({"type": "done"})
            self.wfile.write(f"data: {done_event}\n\n".encode("utf-8"))
            self.wfile.flush()

        except BrokenPipeError:
            # Client disconnected / stopped generation
            pass
        except Exception as e:
            try:
                err_event = json.dumps({"type": "error", "error": str(e)})
                self.wfile.write(f"data: {err_event}\n\n".encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass


def run_server(port: int = 8000):
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, OSTutorHandler)
    print("=" * 60)
    print(f"  🎓 OSTutor Web Interface & RAG Server running!")
    print(f"  👉 Open your browser at: http://localhost:{port}")
    print(f"  📁 Serving from: {CURRENT_DIR}")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSTutor Web Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default 8000)")
    args = parser.parse_args()
    run_server(args.port)
