#!/usr/bin/env python3
"""Proof-of-concept counterpart to rank0.py - see that file and
../README.md. Run on the second machine.

Usage: rank1.py [--addr 10.10.10.1] [--port 29500]
"""
import argparse
import torch
import torch.distributed as dist


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--addr", default="10.10.10.1", help="rank 0's address")
    p.add_argument("--port", default="29500")
    p.add_argument("--iface", default="thunderbolt0")
    args = p.parse_args()

    import os

    os.environ["GLOO_SOCKET_IFNAME"] = args.iface

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{args.addr}:{args.port}",
        rank=1,
        world_size=2,
    )

    sample = torch.empty(1, 4, 128, 128)
    dist.recv(sample, src=0)
    print(f"[rank1] received tensor {sample.shape}, echoing back")
    dist.send(sample, dst=0)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
