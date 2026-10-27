#!/usr/bin/env python3
"""Hands out one unclaimed prompt per request, accepts finished images back.

stdlib-only on purpose - see ../README.md for when to swap this for a
real queue (RQ, Celery) instead.

prompts.txt format: one prompt per line. Blank lines and lines starting
with # are skipped.

Usage: coordinator.py --prompts prompts.txt [--port 8765] [--outdir out/]
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class JobQueue:
    def __init__(self, prompts: list[str]):
        self._lock = threading.Lock()
        self._pending = list(enumerate(prompts))
        self._in_flight: dict[int, float] = {}
        self._done: set[int] = set()
        self._requeue_after = 600  # seconds - assume a worker died and retry

    def take(self) -> tuple[int, str] | None:
        with self._lock:
            now = time.time()
            # requeue anything that's been out too long without a result
            stale = [i for i, t in self._in_flight.items() if now - t > self._requeue_after]
            for i in stale:
                del self._in_flight[i]
                self._pending.append((i, self._prompt_text(i)))

            if not self._pending:
                return None
            idx, prompt = self._pending.pop(0)
            self._in_flight[idx] = now
            return idx, prompt

    def complete(self, idx: int) -> None:
        with self._lock:
            self._in_flight.pop(idx, None)
            self._done.add(idx)

    def _prompt_text(self, idx: int) -> str:
        return self._all_prompts[idx]

    def attach_prompts(self, prompts: list[str]) -> None:
        self._all_prompts = prompts

    def status(self) -> dict:
        with self._lock:
            return {
                "pending": len(self._pending),
                "in_flight": len(self._in_flight),
                "done": len(self._done),
            }


def make_handler(queue: JobQueue, outdir: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/job":
                job = queue.take()
                if job is None:
                    self._send_json({"done": True})
                    return
                idx, prompt = job
                self._send_json({"id": idx, "prompt": prompt})
            elif self.path == "/status":
                self._send_json(queue.status())
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            # /result/<id> with the image bytes as the body
            if not self.path.startswith("/result/"):
                self._send_json({"error": "not found"}, 404)
                return
            idx = int(self.path.rsplit("/", 1)[-1])
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length)
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / f"{idx:04d}.png").write_bytes(data)
            queue.complete(idx)
            self._send_json({"ok": True})

        def log_message(self, fmt, *args):
            print(f"[coordinator] {self.address_string()} {fmt % args}")

    return Handler


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True, type=Path)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--outdir", type=Path, default=Path("out"))
    args = p.parse_args()

    lines = args.prompts.read_text().splitlines()
    prompts = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
    if not prompts:
        raise SystemExit(f"no prompts found in {args.prompts}")

    queue = JobQueue(prompts)
    queue.attach_prompts(prompts)
    print(f"[coordinator] {len(prompts)} jobs queued, serving on :{args.port}")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(queue, args.outdir))
    server.serve_forever()


if __name__ == "__main__":
    main()
