import sys, json, threading, time, urllib.request, urllib.error, tempfile
from pathlib import Path
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fleet"))
import coordinator as c


def test_parse():
    jobs = c.parse_prompts("# hi\n\nplain one\nbox | with pipe | control=a.png\n")
    assert [(j.prompt, j.control) for j in jobs] == [("plain one", None), ("box | with pipe", "a.png")]


def test_requeue_and_late_result():
    q = c.JobQueue(c.parse_prompts("a\nb"), requeue_after=0.05)
    j = q.take(); time.sleep(0.1)
    assert q.status()["pending"] == 2          # a requeued
    assert q.complete(j.id)                    # late result wins
    assert q.status() == {"pending": 1, "in_flight": 0, "done": 1, "failed": []}
    assert not q.complete(j.id)                # duplicate rejected
    assert q.take().id == 1 and q.take() is None


def test_requeued_job_not_handed_twice_after_late_result():
    q = c.JobQueue(c.parse_prompts("a"), requeue_after=0.05)
    j = q.take(); time.sleep(0.1); q.status()
    q.complete(j.id)
    assert q.take() is None


def test_max_attempts():
    q = c.JobQueue(c.parse_prompts("a"), max_attempts=2)
    for _ in range(2):
        q.fail(q.take().id)
    assert q.take() is None and q.status()["failed"] == [0]


def test_http_and_controls():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); (d / "ctl").mkdir(); (d / "ctl" / "a.png").write_bytes(b"PNG!")
        (d / "secret.txt").write_text("no")
        q = c.JobQueue(c.parse_prompts("x | control=a.png\ny"))
        srv = ThreadingHTTPServer(("127.0.0.1", 0), c.make_handler(q, d / "out", d / "ctl", "a.png"))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_port}"
        get = lambda p: json.loads(urllib.request.urlopen(base + p).read())
        assert get("/job") == {"id": 0, "prompt": "x", "control": "a.png"}
        assert get("/job")["control"] == "a.png"   # default applied
        assert urllib.request.urlopen(base + "/control/a.png").read() == b"PNG!"
        for bad in ("/control/../secret.txt", "/control/%2e%2e/secret.txt"):
            try: urllib.request.urlopen(base + bad); assert False
            except urllib.error.HTTPError as e: assert e.code == 404
        post = lambda p: urllib.request.urlopen(urllib.request.Request(base + p, data=b"img", method="POST"))
        post("/result/0")
        assert (d / "out" / "0000.png").read_bytes() == b"img"
        try: post("/result/0"); assert False
        except urllib.error.HTTPError as e: assert e.code == 409
        assert get("/job") == {"done": False}      # job 1 still in flight
        post("/result/1")
        assert get("/job") == {"done": True}
        srv.shutdown()
