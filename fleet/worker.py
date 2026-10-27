#!/usr/bin/env python3
"""Pulls one job at a time from the coordinator, renders it with SD1.5 +
ControlNet, posts the PNG back. See ../README.md.

Each job may name its own control image (fetched from the coordinator and
cached); jobs without one use --control-image.

Requires: diffusers, transformers, accelerate, torch (pinned to match
every other node exactly - see the README's apples-to-apples section).

Usage: worker.py --coordinator http://10.10.10.1:8765 [--control-image plan.png]
"""
import argparse
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image

HTTP_TIMEOUT = 30


def load_pipeline(control_model: str, base_model: str):
    controlnet = ControlNetModel.from_pretrained(control_model, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.enable_model_cpu_offload()  # keeps this workable on 4GB VRAM
    return pipe


def fetch_job(coordinator: str) -> dict:
    """Returns a job dict, or {"done": bool} when nothing is available."""
    with urllib.request.urlopen(f"{coordinator}/job", timeout=HTTP_TIMEOUT) as r:
        return json.loads(r.read())


def fetch_control(coordinator: str, name: str, cache: dict) -> Image.Image:
    if name not in cache:
        url = f"{coordinator}/control/{urllib.parse.quote(name)}"
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as r:
            cache[name] = Image.open(io.BytesIO(r.read())).convert("RGB")
    return cache[name]


def post(coordinator: str, path: str, data: bytes = b"") -> None:
    req = urllib.request.Request(f"{coordinator}{path}", data=data, method="POST")
    try:
        urllib.request.urlopen(req, timeout=HTTP_TIMEOUT).read()
    except urllib.error.HTTPError as e:
        if e.code != 409:  # 409: another worker already delivered this job
            raise


def post_result(coordinator: str, job_id: int, image: Image.Image) -> None:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    post(coordinator, f"/result/{job_id}", buf.getvalue())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--coordinator", required=True, help="e.g. http://10.10.10.1:8765")
    p.add_argument("--control-image", default=None,
                   help="fallback lineart/canny/depth map for jobs that name no control "
                        "image - e.g. a traced plan outline, not a copyrighted photo")
    p.add_argument("--control-model", default="lllyasviel/sd-controlnet-canny")
    p.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--poll-interval", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=25)
    args = p.parse_args()

    fallback = Image.open(args.control_image).convert("RGB") if args.control_image else None
    pipe = load_pipeline(args.control_model, args.base_model)
    cache: dict[str, Image.Image] = {}

    while True:
        try:
            job = fetch_job(args.coordinator)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"[worker] coordinator unreachable ({e}), retrying")
            time.sleep(args.poll_interval)
            continue

        if "id" not in job:
            if job.get("done"):
                print("[worker] all jobs finished, exiting")
                return
            time.sleep(args.poll_interval)
            continue

        print(f"[worker] job {job['id']}: {job['prompt']!r} control={job.get('control')}")
        try:
            if job.get("control"):
                control = fetch_control(args.coordinator, job["control"], cache)
            elif fallback is not None:
                control = fallback
            else:
                raise RuntimeError("job has no control image and no --control-image given")
            result = pipe(job["prompt"], image=control,
                          num_inference_steps=args.steps).images[0]
            post_result(args.coordinator, job["id"], result)
            print(f"[worker] job {job['id']} done")
        except Exception as e:  # report and carry on; coordinator retries the job
            print(f"[worker] job {job['id']} failed: {e!r}")
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            try:
                post(args.coordinator, f"/fail/{job['id']}")
            except (urllib.error.URLError, OSError):
                pass


if __name__ == "__main__":
    main()
