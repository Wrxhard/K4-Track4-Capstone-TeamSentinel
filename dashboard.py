"""Local, read-only dashboard. Inference owns state; HTTP serves snapshots."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Lock, Thread


class Dashboard:
    def __init__(self, port=8080):
        self.lock = Lock()
        self.status = {
            "status": "starting",
            "objects": [],
            "alerts": [],
            "timestamp": 0,
        }
        self.jpeg = b""
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?")[0]
                with dashboard.lock:
                    if path == "/api/status":
                        body, content_type = (
                            json.dumps(dashboard.status, allow_nan=False).encode(),
                            "application/json",
                        )
                    elif path == "/frame.jpg":
                        if not dashboard.jpeg:
                            self.send_error(503, "Waiting for first frame")
                            return
                        body, content_type = dashboard.jpeg, "image/jpeg"
                    elif path == "/":
                        body, content_type = (
                            Path(__file__)
                            .with_name("web")
                            .joinpath("index.html")
                            .read_bytes(),
                            "text/html; charset=utf-8",
                        )
                    else:
                        self.send_error(404)
                        return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def publish(self, status, jpeg=None):
        # Copy via serialization so readers cannot see a mutable inference state.
        with self.lock:
            self.status = json.loads(json.dumps(status, allow_nan=False))
            if jpeg is not None:
                self.jpeg = bytes(jpeg)

    def wait(self):
        try:
            Event().wait()
        except KeyboardInterrupt:
            pass

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
