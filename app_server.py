
"""
app_server.py - OSTutor Frontend & RAG Backend Server
Serves the web UI and provides streaming SSE endpoints connected to rag_service.py.
"""

import argparse
import json
import os
import sys
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
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization"
        )
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
        elif path in ("/", ""):
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
            self.send_error(
                HTTPStatus.NOT_FOUND,
                "Endpoint not found"
            )

    def _read_json_body(self):
        content_length = int(self.headers.get("Content-Length", 0))

        if content_length > 0:
            body = self.rfile.read(content_length).decode("utf-8")
            return json.loads(body)

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
        except Exception as e:
            self._send_json(
                {"error": str(e)},
                HTTPStatus.INTERNAL_SERVER_ERROR
            )

    def handle_api_models(self):
        try:
            self._send_json({
                "models": rag_service.get_available_models()
            })
        except Exception as e:
            self._send_json(
                {"error": str(e)},
                HTTPStatus.INTERNAL_SERVER_ERROR
            )

    def handle_api_ingest(self):
        try:
            result = rag_service.ingest_notes()
            self._send_json(result)
        except Exception as e:
            self._send_json(
                {"error": str(e)},
                HTTPStatus.INTERNAL_SERVER_ERROR
            )

    def handle_chat_stream(self):
        try:
            body = self._read_json_body()
            prompt = body.get("prompt", "").strip()
            mode = body.get("mode", "rag")
            model = body.get("model") or None

            if not prompt:
                self.send_error(
                    HTTPStatus.BAD_REQUEST,
                    "Missing prompt in request body"
                )
                return

            if mode == "rag":
                sources, token_stream = rag_service.generate_rag(
                    prompt, model=model
                )
            elif mode == "plain":
                sources, token_stream = rag_service.generate_plain(
                    prompt, model=model
                )
            elif mode == "mcq":
                sources, token_stream = rag_service.generate_mcq(
                    prompt, model=model
                )
            elif mode == "viva":
                sources, token_stream = rag_service.generate_viva(
                    prompt, model=model
                )
            else:
                sources, token_stream = rag_service.generate_rag(
                    prompt, model=model
                )

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            init_event = json.dumps({
                "type": "sources",
                "sources": sources
            })
            self.wfile.write(f"data: {init_event}\n\n".encode("utf-8"))
            self.wfile.flush()

            for token in token_stream:
                token_event = json.dumps({
                    "type": "token",
                    "token": token
                })
                self.wfile.write(
                    f"data: {token_event}\n\n".encode("utf-8")
                )
                self.wfile.flush()

            done_event = json.dumps({"type": "done"})
            self.wfile.write(f"data: {done_event}\n\n".encode("utf-8"))
            self.wfile.flush()

        except BrokenPipeError:
            pass
        except Exception as e:
            try:
                err_event = json.dumps({
                    "type": "error",
                    "error": str(e)
                })
                self.wfile.write(
                    f"data: {err_event}\n\n".encode("utf-8")
                )
                self.wfile.flush()
            except Exception:
                pass


def run_server(port=8000):
    server_address = ("0.0.0.0", port)
    httpd = ThreadingHTTPServer(server_address, OSTutorHandler)

    print(f"OSTutor server running on port {port}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSTutor Web Server")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to listen on"
    )
    args = parser.parse_args()

    port = args.port or int(os.environ.get("PORT", 8000))
    run_server(port)
