"""Builds the SD1.5 + ControlNet pipeline from a Config, shared by worker.py
and bench.py so the benchmark measures exactly what the worker runs.

A Config round-trips through JSON: bench.py writes the winner to
bench-best.json and `worker.py --config bench-best.json` loads it.
"""
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from diffusers import (ControlNetModel, DPMSolverMultistepScheduler,
                       StableDiffusionControlNetPipeline, UniPCMultistepScheduler)
from PIL import Image

DTYPES = {"fp16": torch.float16, "fp32": torch.float32}
SCHEDULERS = {"unipc": UniPCMultistepScheduler, "dpmpp": DPMSolverMultistepScheduler}


@dataclass(frozen=True)
class Config:
    dtype: str = "fp32"        # fp16 | fp32 (fp16 gives NaN/black images on GTX 16xx, slow on 10xx)
    offload: str = "sequential"  # none | model | sequential (the safe default fits any GPU)
    attention: str = "sdpa"    # sdpa | slicing
    vae: str = "normal"        # normal | slicing | tiling
    channels_last: bool = True
    scheduler: str = "dpmpp"   # default | unipc | dpmpp
    steps: int = 12
    size: int = 512

    def label(self) -> str:
        parts = [self.dtype, f"off-{self.offload}", self.attention, f"vae-{self.vae}"]
        if self.channels_last:
            parts.append("chlast")
        parts += [self.scheduler, f"{self.steps}st", f"{self.size}px"]
        return "_".join(parts)

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Config":
        return cls(**json.loads(Path(path).read_text()))


def build_pipeline(cfg: Config, control_model: str, base_model: str):
    dt = DTYPES[cfg.dtype]
    controlnet = ControlNetModel.from_pretrained(control_model, torch_dtype=dt)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model, controlnet=controlnet, torch_dtype=dt, safety_checker=None
    )
    if cfg.scheduler != "default":
        pipe.scheduler = SCHEDULERS[cfg.scheduler].from_config(pipe.scheduler.config)
    if cfg.attention == "slicing":
        pipe.enable_attention_slicing()
    if cfg.vae == "slicing":
        pipe.vae.enable_slicing()
    elif cfg.vae == "tiling":
        pipe.vae.enable_tiling()
    if cfg.channels_last:
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.controlnet.to(memory_format=torch.channels_last)
    if cfg.offload == "none":
        pipe.to("cuda")
    elif cfg.offload == "model":
        pipe.enable_model_cpu_offload()
    else:
        pipe.enable_sequential_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def is_blank(image: Image.Image) -> bool:
    """True for an all-black/flat image - the signature of fp16 NaNs."""
    lo, hi = image.convert("L").getextrema()
    return hi - lo < 8


def render(pipe, cfg: Config, prompt: str, control: Image.Image, seed: int | None = None):
    control = control.resize((cfg.size, cfg.size))
    gen = torch.Generator("cpu").manual_seed(seed) if seed is not None else None
    return pipe(prompt, image=control, num_inference_steps=cfg.steps,
                height=cfg.size, width=cfg.size, generator=gen).images[0]
