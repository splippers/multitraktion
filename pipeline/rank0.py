#!/usr/bin/env python3
"""Proof-of-concept: establish a torch.distributed link over Thunderbolt
networking and prove a tensor crosses it. Not a model split yet - see
../README.md for what to build once this works.

Run on the machine holding the static IP used as MASTER_ADDR (node 0).
Pair with rank1.py on the other machine.

Usage: rank0.py [--addr 10.10.10.1] [--port 29500]
"""
import argparse
import torch
import torch.distributed as dist


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--addr", default="10.10.10.1")
    p.add_argument("--port", default="29500")
    p.add_argument("--iface", default="thunderbolt0")
    args = p.parse_args()

    import os

    os.environ["GLOO_SOCKET_IFNAME"] = args.iface

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{args.addr}:{args.port}",
        rank=0,
        world_size=2,
    )

    # Send a tensor, time the round trip. Swap this for a real latent
    # tensor (shape/dtype matching your pipeline) once you're profiling
    # the actual split, not just proving the link works.
    sample = torch.randn(1, 4, 128, 128)
    print(f"[rank0] sending tensor {sample.shape}")
    dist.send(sample, dst=1)

    echoed = torch.empty_like(sample)
    dist.recv(echoed, src=1)
    assert torch.allclose(sample, echoed), "round trip mismatch - check the link"
    print("[rank0] round trip OK")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
