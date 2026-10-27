#!/usr/bin/env python3
"""Finds the fastest worker configuration for this node's GPU.

Each candidate config runs in its own subprocess (so an out-of-memory
failure or leaked VRAM can't poison the next one), renders one untimed
warm-up image, then N timed images with a fixed seed while sampling
nvidia-smi for utilisation, SM clock, temperature and power.

Stages (greedy - each keeps a knob only if it is >=3% faster):
  1. dtype x offload   - fp16 vs fp32, resident vs offloaded weights
  2. memory/speed knobs - attention slicing, VAE slicing/tiling, channels_last
  3. (--tune-quality) scheduler / steps / size. These change the picture, so
     they never go in bench-best.json; the fastest goes in bench-fast.json and
     every image is saved to bench-out/ for you to eyeball.

Writes bench-results.json, bench-best.json (same output quality as the
baseline), and prints the worker command to use.

Usage: bench.py [--images 3] [--tune-quality] [--control-image plan.png]
"""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sdpipe import Config  # noqa: E402

PROMPT = "a Victorian brick signal box with slate roof, concept illustration, soft light"
GAIN = 0.97  # a knob must beat the current best by 3% to be adopted


# ---------------------------------------------------------------- child ----

class GpuSampler:
    """Polls nvidia-smi in the background while a config renders."""
    FIELDS = "utilization.gpu,clocks.sm,temperature.gpu,power.draw"

    def __init__(self):
        self.rows: list[list[float]] = []
        self._proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits",
             "-lms", "500"], stdout=subprocess.PIPE, text=True)
        self._t = threading.Thread(target=self._read, daemon=True)
        self._t.start()

    def _read(self):
        for line in self._proc.stdout:
            try:
                self.rows.append([float(x) for x in line.split(",")])
            except ValueError:  # "[N/A]" on some mobile parts
                pass

    def stop(self) -> dict:
        self._proc.terminate()
        self._t.join(timeout=2)
        if not self.rows:
            return {}
        col = list(zip(*self.rows))
        return {"gpu_util_pct": round(statistics.mean(col[0])),
                "sm_clock_mhz": round(statistics.mean(col[1])),
                "max_temp_c": max(col[2]), "avg_power_w": round(statistics.mean(col[3]), 1)}


def run_one(cfg: Config, args) -> dict:
    import torch
    from PIL import Image
    from sdpipe import build_pipeline, render

    control = Image.open(args.control_image).convert("RGB")
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    try:
        t0 = time.time()
        pipe = build_pipeline(cfg, args.control_model, args.base_model)
        load_s = time.time() - t0
        render(pipe, cfg, args.prompt, control, seed=0)  # warm-up (cudnn autotune, allocator)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        sampler = GpuSampler()
        times = []
        for _ in range(args.images):
            t = time.time()
            img = render(pipe, cfg, args.prompt, control, seed=0)
            torch.cuda.synchronize()
            times.append(time.time() - t)
        stats = sampler.stop()
        img.save(out / f"{cfg.label()}.png")
        med = statistics.median(times)
        return {"ok": True, "s_per_img": round(med, 2), "s_per_step": round(med / cfg.steps, 3),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "load_s": round(load_s, 1), **stats}
    except torch.cuda.OutOfMemoryError:
        return {"ok": False, "error": "out of VRAM"}
    except Exception as e:
        return {"ok": False, "error": repr(e)[:200]}


# --------------------------------------------------------------- parent ----

def measure(cfg: Config, args, cache: dict) -> dict:
    if cfg in cache:
        return cache[cfg]
    print(f"  {cfg.label():<70}", end="", flush=True)
    cmd = [sys.executable, __file__, "--one", json.dumps(cfg.__dict__)] + args.passthrough
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        line = next((l for l in reversed(p.stdout.splitlines()) if l.startswith("RESULT ")), None)
        res = json.loads(line[7:]) if line else {"ok": False, "error": (p.stderr or "no output")[-200:]}
    except subprocess.TimeoutExpired:
        res = {"ok": False, "error": f"timeout >{args.timeout}s"}
    cache[cfg] = res
    if res["ok"]:
        print(f"{res['s_per_img']:>7.2f} s/img  {res['peak_vram_gb']:>5.2f} GB  "
              f"util {res.get('gpu_util_pct', '?')}%  {res.get('sm_clock_mhz', '?')} MHz  "
              f"{res.get('max_temp_c', '?')}C")
    else:
        print(f"FAILED: {res['error']}")
    return res


def best_of(cands: list[Config], args, cache) -> tuple[Config | None, dict | None]:
    scored = [(measure(c, args, cache), c) for c in cands]
    ok = [(r, c) for r, c in scored if r["ok"]]
    if not ok:
        return None, None
    r, c = min(ok, key=lambda rc: rc[0]["s_per_img"])
    return c, r


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--one", help=argparse.SUPPRESS)
    p.add_argument("--images", type=int, default=3, help="timed images per config")
    p.add_argument("--control-image", type=Path,
                   default=HERE.parent / "controls" / "plan.png")
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--control-model", default="lllyasviel/sd-controlnet-canny")
    p.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--out", default="bench-out")
    p.add_argument("--timeout", type=int, default=900, help="seconds per config")
    p.add_argument("--tune-quality", action="store_true",
                   help="also sweep scheduler/steps/size (changes the image)")
    args = p.parse_args()

    if args.one:  # child process
        res = run_one(Config(**json.loads(args.one)), args)
        print("RESULT " + json.dumps(res))
        return

    args.passthrough = ["--images", str(args.images), "--control-image", str(args.control_image),
                        "--prompt", args.prompt, "--control-model", args.control_model,
                        "--base-model", args.base_model, "--out", args.out]
    if not args.control_image.exists():
        raise SystemExit(f"control image not found: {args.control_image} (pass --control-image)")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(f"GPU: {gpu}\nTimed images per config: {args.images} (after 1 warm-up)\n")

    cache: dict[Config, dict] = {}
    base = Config()

    print("Stage 1: dtype x offload")
    stage1 = [base.with_(dtype=d, offload=o)
              for d in ("fp16", "fp32") for o in ("none", "model", "sequential")]
    best, res = best_of(stage1, args, cache)
    if best is None:
        raise SystemExit("\nEvery stage-1 config failed - see errors above "
                         "(driver/CUDA problem, or the model downloads failed).")

    print("\nStage 2: memory/speed knobs")
    rescue = [base.with_(dtype=d, offload="none", attention="slicing", vae="slicing")
              for d in ("fp16", "fp32")
              if not cache[base.with_(dtype=d, offload="none")]["ok"]]
    if rescue:  # resident weights OOM'd - slicing may squeeze them in
        c, r = best_of(rescue, args, cache)
        if c and r["s_per_img"] < res["s_per_img"] * GAIN:
            best, res = c, r
    for knob, val in (("attention", "slicing"), ("vae", "slicing"), ("vae", "tiling"),
                      ("channels_last", True)):
        if getattr(best, knob) == val:
            continue
        cand = best.with_(**{knob: val})
        r = measure(cand, args, cache)
        if r["ok"] and r["s_per_img"] < res["s_per_img"] * GAIN:
            best, res = cand, r

    best.save(Path("bench-best.json"))
    fast = best
    if args.tune_quality:
        print("\nStage 3: quality-affecting (eyeball bench-out/ before using)")
        cands = [best.with_(scheduler=s, steps=n)
                 for s in ("unipc", "dpmpp") for n in (12, 16, 20)]
        cands += [best.with_(size=s) for s in (384, 448)]
        c, r = best_of(cands, args, cache)
        if c and r["s_per_img"] < res["s_per_img"]:
            fast = c
            fast.save(Path("bench-fast.json"))

    Path("bench-results.json").write_text(json.dumps(
        [{"config": c.__dict__, **r} for c, r in cache.items()], indent=2) + "\n")

    base_r = cache[base.with_(offload="model")]
    print("\n=== Result ===")
    print(f"Best (same image quality): {best.label()}")
    print(f"  {res['s_per_img']} s/img, {res['peak_vram_gb']} GB VRAM, "
          f"GPU util {res.get('gpu_util_pct', '?')}%, {res.get('sm_clock_mhz', '?')} MHz, "
          f"{res.get('max_temp_c', '?')}C")
    if base_r["ok"]:
        print(f"  vs current worker default: {base_r['s_per_img']} s/img "
              f"-> {base_r['s_per_img'] / res['s_per_img']:.2f}x")
    util = res.get("gpu_util_pct")
    if util is not None and util < 70:
        print(f"  GPU only {util}% busy: still waiting on weight transfers or the CPU.")
    if res.get("sm_clock_mhz") and res.get("max_temp_c", 0) >= 85:
        print("  Hot (>=85C): the GPU is probably thermal-throttling - check the SM clock "
              "against the card's boost clock and try a cooling pad / fresh paste.")
    print("\nUse it:  python3 fleet/worker.py --coordinator <url> --config bench-best.json")
    if fast != best:
        print(f"Faster (lower quality, check bench-out/): {fast.label()}\n"
              f"         ... --config bench-fast.json")


if __name__ == "__main__":
    main()
