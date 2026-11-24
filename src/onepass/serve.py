"""A local System One server: POST /v1/systemone answers typed questions, and / serves a demo page.

    python -m onepass.serve path/to/model.gguf --port 8000
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files

from onepass.model import Model


def make_handler(model, lock, page):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/health":
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                state, questions = body["state"], body["questions"]
                if not isinstance(questions, dict) or not questions:
                    raise ValueError("questions must be a non-empty object")
            except KeyError as e:
                return self._send(400, {"error": f"missing field {e}; send {{\"state\": ..., \"questions\": {{...}}}}"})
            except (ValueError, TypeError) as e:
                return self._send(400, {"error": f"bad request: {e}"})
            try:
                with lock:  # one llama.cpp context, one request at a time
                    t = time.perf_counter()
                    answers = model.ask(state, questions)
                    ms = (time.perf_counter() - t) * 1000
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            self._send(200, {"answers": answers, "latency_ms": round(ms, 1), "generated_tokens": 0})

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", help="GGUF file")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--temperature", type=float, default=1.0, help="calibration temperature from training")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--gpu-layers", type=int, default=-1, help="-1 puts every layer on the GPU, 0 runs on the CPU")
    args = ap.parse_args()
    model = Model(args.model, temperature=args.temperature, threads=args.threads, gpu_layers=args.gpu_layers)
    page = files("onepass").joinpath("demo.html").read_bytes()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model, threading.Lock(), page))
    print(f"onepass serving {args.model} on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        model.close()


if __name__ == "__main__":
    main()
