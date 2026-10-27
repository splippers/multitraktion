#!/usr/bin/env python3
"""Pulls one job at a time from the coordinator, renders it with SD1.5 +
ControlNet, posts the PNG back. See ../README.md.

Requires: diffusers, transformers, accelerate, torch (pinned to match
every other node exactly - see the README's apples-to-apples section).

Usage: worker.py --coordinator http://10.10.10.1:8765 --control-image plan.png
"""
import argparse
import io
import time
import urllib.request

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image


def load_pipeline(control_model: str, base_model: str):
    controlnet = ControlNetModel.from_pretrained(control_model, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.enable_model_cpu_offload()  # keeps this workable on 4GB VRAM
    return pipe


def fetch_job(coordinator: str) -> dict | None:
    with urllib.request.urlopen(f"{coordinator}/job") as r:
        import json

        job = json.loads(r.read())
        return None if job.get("done") else job


def post_result(coordinator: str, job_id: int, image: Image.Image) -> None:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    req = urllib.request.Request(
        f"{coordinator}/result/{job_id}", data=buf.getvalue(), method="POST"
    )
    urllib.request.urlopen(req).read()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--coordinator", required=True, help="e.g. http://10.10.10.1:8765")
    p.add_argument("--control-image", required=True,
                    help="lineart/canny/depth map to constrain output - e.g. a traced "
                         "plan outline or footprint, not a copyrighted reference photo")
    p.add_argument("--control-model", default="lllyasviel/sd-controlnet-canny")
    p.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--poll-interval", type=float, default=5.0)
    args = p.parse_args()

    pipe = load_pipeline(args.control_model, args.base_model)
    control_image = Image.open(args.control_image).convert("RGB")

    while True:
        job = fetch_job(args.coordinator)
        if job is None:
            print("[worker] queue empty, waiting")
            time.sleep(args.poll_interval)
            continue

        print(f"[worker] job {job['id']}: {job['prompt']!r}")
        result = pipe(job["prompt"], image=control_image).images[0]
        post_result(args.coordinator, job["id"], result)
        print(f"[worker] job {job['id']} done")


if __name__ == "__main__":
    main()
