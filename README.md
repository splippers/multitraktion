# MultiTraktion

*or, properly,*
**Schmalspurmehrfachtraktionsrechenclustermultiplexskalierungsverbund** —
narrow-gauge-multiple-traction-computing-cluster-multiplex-scaling-federation,
if you're Germanic about it.

`Mehrfachtraktion` is the real German railway term for coupling several
locomotives together and running them as one unit under one driver's
control — exactly what this does with GPUs too weak to pull the train
alone. Turns two or more identical low-VRAM laptops (the reference build:
Dell XPS 15 9570, 2018, 32GB RAM, 4GB VRAM GTX 1050 Ti) into either a
render fleet or, for the more ambitious build, a single pipeline-parallel
model split across their GPUs.

Spun out of [The Permanent Way](https://github.com/splippers/thepermanentway),
a railway game needing structure illustrations generated from survey
material, but nothing here is specific to that project. Point it at
whatever prompts and control images you like.

Two tiers, in order of effort:

1. **Fleet (`fleet/`)** — each machine runs a full SD1.5 + ControlNet
   pipeline independently. No model splitting, no VRAM pooling. N
   identical laptops = N× throughput. Works today, no distributed-systems
   code required beyond a trivial job queue.
2. **Pipeline (`pipeline/`)** — a model too big for one card (SDXL needs
   ~6.5–8GB; these cards have 4GB) is split across machines, each holding
   part of the UNet. Genuinely interesting, genuinely more work, and
   bottlenecked by whatever link joins the machines.

Start with the fleet. Only build the pipeline if you specifically want
SDXL-quality output or want the distributed-systems exercise for its own
sake — it will not make four 4GB cards behave like one 16GB card.

## Apples-to-apples build (do this before any of the above)

Hardware identical is not the same as software identical. Lock down one
reference machine first, then match every other node to it exactly:

```
lsb_release -a
nvidia-smi          # driver version + CUDA version, from the header
python3 --version
```

Install the *same* Ubuntu point release and the *same* NVIDIA driver
version on every node — not "whatever apt currently offers," which drifts
over time. Pull the exact driver from NVIDIA's archive if apt has moved
on. Pin every Python package to the same version across all nodes:

```
python3 -m venv ~/sdvenv && source ~/sdvenv/bin/activate
pip install torch==<pinned> --index-url https://download.pytorch.org/whl/cu<matching cuda>
pip install diffusers==<pinned> transformers==<pinned> accelerate==<pinned>
```

Record the exact versions once the reference machine is set up and reuse
them verbatim on every other node.

## Networking: 2 nodes vs 4+

**Two nodes:** Thunderbolt networking over a single TB3-certified cable
(plain USB-C data cables will not negotiate it) is the best price/latency
trade-off — no extra hardware, ~10–20Gbps real throughput, lowest latency
option short of PCIe.

```
boltctl list                 # see the other device
boltctl enroll <uuid>        # authorize, both directions
ip link show                 # find the new interface, e.g. thunderbolt0
sudo ip addr add 10.10.10.1/24 dev thunderbolt0   # node 0
sudo ip addr add 10.10.10.2/24 dev thunderbolt0   # node 1
sudo ip link set thunderbolt0 up                  # both

# verify before trusting it:
iperf3 -s                    # node 0
iperf3 -c 10.10.10.1          # node 1 — expect multi-Gbps, not ~5Gbps (that means
                               # it fell back to plain USB networking, not Thunderbolt)
```

**Four or more nodes:** Thunderbolt networking is point-to-point between
two hosts; it does not cleanly extend to a 4-way mesh through a dock or
hub. For the 4-laptop build, use a small switch instead — a cheap
2.5GbE (or 10GbE, if the budget stretches) switch plus a USB-C/TB3 2.5GbE
adapter per laptop. Slower than the TB3 point-to-point link, still well
ahead of plain gigabit Ethernet or Wi-Fi, and it scales to any number of
nodes without topology headaches. Use the same `iperf3` check per pair of
nodes before trusting the network.

## Fleet (recommended default)

`fleet/coordinator.py` holds a queue of prompts (one per subject, however
you've organised them for your own project) and hands one unclaimed job
to each worker that asks. `fleet/worker.py` runs SD1.5 + ControlNet
locally, generates the image, and uploads the result back.

```
# coordinator (any one node):
python3 fleet/coordinator.py --prompts prompts.txt --port 8765

# every worker, including the coordinator's own machine if it has a GPU:
python3 fleet/worker.py --coordinator http://10.10.10.1:8765 \
    --control-image path/to/control_image.png
```

Deliberately stdlib-only on the coordinator side (`http.server`) so
nobody has to stand up Redis or similar just to try this out. Swap in a
real queue (RQ, Celery) if the fleet grows past a handful of nodes.

## Pipeline (the 4-GPU stretch goal)

`pipeline/rank0.py` and `pipeline/rank1.py` are a proof-of-concept, not a
working model split — they establish a `torch.distributed` (Gloo) link
over the Thunderbolt interface and prove a tensor can be sent and
received across it before you invest in anything harder. The realistic
first real split, once that's proven, is **not** a layer-by-layer UNet
carve-up (real surgery on diffusers internals) but:

- rank 0: text encoder(s) + VAE decode
- rank 1: the UNet denoising loop

One latent tensor crosses the network per image, not per step, which
keeps the network overhead negligible regardless of link speed. Only
attempt splitting the UNet itself mid-block across ranks — the thing that
would actually let 4×4GB behave more like 16GB for a single SDXL job —
after that coarse split is working and you've profiled where SDXL's UNet
actually concentrates its memory use on this specific card. Expect this
part to be a real project, not an afternoon.

```python
# skeleton, both ranks:
import os, torch.distributed as dist
os.environ["GLOO_SOCKET_IFNAME"] = "thunderbolt0"
dist.init_process_group(backend="gloo",
                         init_method="tcp://10.10.10.1:29500",
                         rank=<0 or 1>, world_size=2)
```

## Contributing

Written to be reproducible on anyone's set of identical low-VRAM laptops,
not specific to one machine or one project. If you build it out further
(the 2.5GbE switch version, the coarse rank0/rank1 split actually
generating images, or the full UNet pipeline split), a PR with what you
found — especially real `iperf3` numbers and per-step timings — is very
welcome.
