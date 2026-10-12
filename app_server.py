
"""
app_server.py - OSTutor Frontend & RAG Backend Server
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
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization"
        )
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, format_string, *args):
        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"{self.client_address[0]} - "
            f"{format_string % args}"
        )

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
            print(f"[OSTutor] Status error: {e}")
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
            print(f"[OSTutor] Models error: {e}")
            self._send_json(
                {"error": str(e)},
                HTTPStatus.INTERNAL_SERVER_ERROR
            )

    def handle_api_ingest(self):
        try:
            start = time.perf_counter()
            print("[OSTutor] Starting note ingestion...")

            result = rag_service.ingest_notes()

            print(
                f"[OSTutor] Ingestion completed in "
                f"{time.perf_counter() - start:.2f}s"
            )
            self._send_json(result)

        except Exception as e:
            print(f"[OSTutor] Ingestion error: {e}")
            traceback.print_exc()
            self._send_json(
                {"error": str(e)},
                HTTPStatus.INTERNAL_SERVER_ERROR
            )

    def handle_chat_stream(self):
        start = time.perf_counter()
        print("\n[OSTutor] New chat request received")

        headers_sent = False

        try:
            body = self._read_json_body()
            prompt = body.get("prompt", "").strip()
            mode = body.get("mode", "rag")
            model = body.get("model") or None

            if not prompt:
                self._send_json(
                    {"error": "Missing prompt in request body"},
                    HTTPStatus.BAD_REQUEST
                )
                return

            print(f"[OSTutor] Mode: {mode}")
            print(f"[OSTutor] Model: {model or 'default'}")

            print("[OSTutor] Preparing generation...")
            generation_start = time.perf_counter()

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

            print(
                f"[OSTutor] Generation setup took "
                f"{time.perf_counter() - generation_start:.2f}s"
            )

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            headers_sent = True

            init_event = json.dumps({
                "type": "sources",
                "sources": sources
            })

            self.wfile.write(
                f"data: {init_event}\n\n".encode("utf-8")
            )
            self.wfile.flush()

            first_token = True
            token_count = 0

            for token in token_stream:
                if first_token:
                    print(
                        f"[OSTutor] First token arrived after "
                        f"{time.perf_counter() - start:.2f}s"
                    )
                    first_token = False

                token_event = json.dumps({
                    "type": "token",
                    "token": token
                })

                self.wfile.write(
                    f"data: {token_event}\n\n".encode("utf-8")
                )
                self.wfile.flush()
                token_count += 1

            done_event = json.dumps({"type": "done"})
            self.wfile.write(
                f"data: {done_event}\n\n".encode("utf-8")
            )
            self.wfile.flush()

            print(
                f"[OSTutor] Completed in "
                f"{time.perf_counter() - start:.2f}s "
                f"({token_count} tokens)"
            )

        except (BrokenPipeError, ConnectionResetError):
            print("[OSTutor] Client disconnected during streaming")

        except Exception as e:
            print(f"[OSTutor] Chat error: {e}")
            traceback.print_exc()

            if headers_sent:
                try:
                    err_event = json.dumps({
                        "type": "error",
                        "error": str(e)
                    })
                    self.wfile.write(
                        f"data: {err_event}\n\n".encode("utf-8")
                    )
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            else:
                try:
                    self._send_json(
                        {"error": str(e)},
                        HTTPStatus.INTERNAL_SERVER_ERROR
                    )
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass


def run_server(port=8000):
    server_address = ("0.0.0.0", port)
    httpd = ThreadingHTTPServer(server_address, OSTutorHandler)

    print(f"OSTutor server running on port {port}")
    print(f"Open http://127.0.0.1:{port} in your browser")
    print("Press Ctrl+C to stop the server.")

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
