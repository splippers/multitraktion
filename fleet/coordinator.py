#!/usr/bin/env python3
"""Hands out one unclaimed prompt per request, accepts finished images back,
and serves per-job control images to workers.

stdlib-only on purpose - see ../README.md for when to swap this for a
real queue (RQ, Celery) instead.

prompts.txt format: one job per line. Blank lines and lines starting with
# are skipped. A line may end with " | control=<file>" to give that job its
own control image, looked up in --control-dir:

    a Victorian signal box, brick, slate roof | control=signalbox.png
    a plain goods shed

Jobs with no control= fall back to --default-control (a file in
--control-dir), or to the worker's own --control-image if neither is set.

Usage: coordinator.py --prompts prompts.txt [--port 8765] [--outdir out/]
                      [--control-dir controls/] [--default-control plan.png]
"""
import argparse
import json
import re
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CONTROL_RE = re.compile(r"^(?P<prompt>.*?)\s+\|\s+control=(?P<control>\S+)$")


@dataclass
class Job:
    id: int
    prompt: str
    control: str | None = None
    attempts: int = 0


def parse_prompts(text: str) -> list[Job]:
    jobs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = CONTROL_RE.match(line)
        prompt, control = (m["prompt"], m["control"]) if m else (line, None)
        jobs.append(Job(id=len(jobs), prompt=prompt, control=control))
    return jobs


class JobQueue:
    """Every job is pending, in flight, or done - never more than one.

    The first result for a job wins. A late result from a worker whose job
    was requeued still completes it (and pulls it back out of pending or
    in-flight), so nothing renders twice unless a worker really did stall.
    """

    def __init__(self, jobs: list[Job], requeue_after: float = 600.0,
                 max_attempts: int = 3):
        self._lock = threading.Lock()
        self._jobs = {j.id: j for j in jobs}
        self._pending: list[int] = [j.id for j in jobs]
        self._in_flight: dict[int, float] = {}
        self._done: set[int] = set()
        self._failed: set[int] = set()
        self._requeue_after = requeue_after
        self._max_attempts = max_attempts

    def _reap(self, now: float) -> None:
        for i in [i for i, t in self._in_flight.items() if now - t > self._requeue_after]:
            self._release(i)

    def _release(self, idx: int) -> None:
        """Put an in-flight job back, or give up on it if it keeps failing."""
        self._in_flight.pop(idx, None)
        if idx in self._done or idx in self._pending:
            return
        if self._jobs[idx].attempts >= self._max_attempts:
            self._failed.add(idx)
        else:
            self._pending.append(idx)

    def take(self) -> Job | None:
        with self._lock:
            now = time.time()
            self._reap(now)
            if not self._pending:
                return None
            job = self._jobs[self._pending.pop(0)]
            job.attempts += 1
            self._in_flight[job.id] = now
            return job

    def complete(self, idx: int) -> bool:
        """Returns False for unknown ids and for jobs already completed."""
        with self._lock:
            if idx not in self._jobs or idx in self._done:
                return False
            self._in_flight.pop(idx, None)
            if idx in self._pending:
                self._pending.remove(idx)
            self._failed.discard(idx)
            self._done.add(idx)
            return True

    def fail(self, idx: int) -> None:
        """A worker reporting it could not render the job (OOM, bad control)."""
        with self._lock:
            if idx in self._in_flight:
                self._release(idx)

    def status(self) -> dict:
        with self._lock:
            self._reap(time.time())
            return {
                "pending": len(self._pending),
                "in_flight": len(self._in_flight),
                "done": len(self._done),
                "failed": sorted(self._failed),
            }


def safe_child(base: Path, name: str) -> Path | None:
    """Resolve name inside base, refusing anything that escapes it."""
    path = (base / name).resolve()
    if base.resolve() in path.parents and path.is_file():
        return path
    return None


def make_handler(queue: JobQueue, outdir: Path, control_dir: Path | None,
                 default_control: str | None):
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
                    self._send_json({"done": queue.status()["in_flight"] == 0})
                    return
                self._send_json({
                    "id": job.id,
                    "prompt": job.prompt,
                    "control": job.control or default_control,
                })
            elif self.path == "/status":
                self._send_json(queue.status())
            elif self.path.startswith("/control/") and control_dir:
                path = safe_child(control_dir, self.path[len("/control/"):])
                if path is None:
                    self._send_json({"error": "not found"}, 404)
                    return
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            m = re.fullmatch(r"/(result|fail)/(\d+)", self.path)
            if not m:
                self._send_json({"error": "not found"}, 404)
                return
            kind, idx = m[1], int(m[2])
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length)
            if kind == "fail":
                queue.fail(idx)
                self._send_json({"ok": True})
                return
            if not queue.complete(idx):
                # unknown id, or someone else already delivered this job
                self._send_json({"ok": False, "error": "unknown or already done"}, 409)
                return
            outdir.mkdir(parents=True, exist_ok=True)
            tmp = outdir / f".{idx:04d}.tmp"
            tmp.write_bytes(data)
            tmp.replace(outdir / f"{idx:04d}.png")
            self._send_json({"ok": True})

        def log_message(self, fmt, *args):
            print(f"[coordinator] {self.address_string()} {fmt % args}")

    return Handler


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True, type=Path)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--outdir", type=Path, default=Path("out"))
    p.add_argument("--control-dir", type=Path, default=None,
                   help="directory of control images workers may fetch")
    p.add_argument("--default-control", default=None,
                   help="file in --control-dir used by jobs without control=")
    p.add_argument("--requeue-after", type=float, default=600.0)
    args = p.parse_args()

    jobs = parse_prompts(args.prompts.read_text())
    if not jobs:
        raise SystemExit(f"no prompts found in {args.prompts}")
    if args.control_dir:
        missing = {j.control or args.default_control for j in jobs} - {None}
        missing = [c for c in missing if safe_child(args.control_dir, c) is None]
        if missing:
            raise SystemExit(f"control images not found in {args.control_dir}: {missing}")

    queue = JobQueue(jobs, requeue_after=args.requeue_after)
    print(f"[coordinator] {len(jobs)} jobs queued, serving on :{args.port}")

    server = ThreadingHTTPServer(
        ("0.0.0.0", args.port),
        make_handler(queue, args.outdir, args.control_dir, args.default_control),
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
