# MultiTraktion

*or, properly,*
**Schmalspurmehrfachtraktionsrechenclustermultiplexskalierungsverbund** —
narrow-gauge-multiple-traction-computing-cluster-multiplex-scaling-federation,
if you're Germanic about it.

`Mehrfachtraktion` is the real German railway term for coupling several
locomotives together and running them as one unit under one driver's
control — exactly what this does with GPUs too weak to pull the train
alone. Turns two or more low-VRAM laptops into either a
render fleet or, for the more ambitious build, a single pipeline-parallel
model split across their GPUs.

Spun out of [The Permanent Way](https://github.com/splippers/thepermanentway),
a railway game needing structure illustrations generated from survey
material, but nothing here is specific to that project. Point it at
whatever prompts and control images you like.

Two tiers, in order of effort:

1. **Fleet (`fleet/`)** — each machine runs a full SD1.5 + ControlNet
   pipeline independently. No model splitting, no VRAM pooling. N
   laptops = roughly N× throughput (they need not be identical: a faster node simply pulls more jobs). Works today, no distributed-systems
   code required beyond a trivial job queue.
2. **Pipeline (`pipeline/`)** — a model too big for one card (SDXL needs
   ~6.5–8GB; these cards have 4GB) is split across machines, each holding
   part of the UNet. Genuinely interesting, genuinely more work, and
   bottlenecked by whatever link joins the machines.

Start with the fleet. Only build the pipeline if you specifically want
SDXL-quality output or want the distributed-systems exercise for its own
sake — it will not make four 4GB cards behave like one 16GB card.

## Reference hardware

| Node | Laptop | GPU (4GB) | NVIDIA driver | Measured, 512px, fp32 + sequential offload |
|---|---|---|---|---|
| LAVE (coordinator) | XPS 15 7590, i7-9750H, 30GB | GTX 1650 Mobile (Turing) | 615 (open) | 25 steps 60.7 s, DPM++ 12 steps 28.3 s |
| diso | XPS 15 9570, i7-8750H, 15GB | GTX 1050 Ti Max-Q (Pascal) | **580** (Pascal needs the proprietary 580 branch; 590+ dropped it) | 25 steps ~82 s, DPM++ 12 steps ~49 s |

The two nodes differ, so tune each one (see "Tuning a node"). Numbers drift
upward over a long run on the 9570 (power-capped Max-Q part), so treat them as rough.
Install the server without a desktop: GNOME/GDM applies its own sleep rules and
overrides logind's lid setting (`deploy/never-sleep-on-ac.sh`).

## Apples-to-apples build (do this before any of the above)

*Needed for the pipeline tier. The fleet tier only needs each node to be internally consistent.*

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
# every node, once (Ubuntu): venv + pinned packages + fingerprint to compare
./setup-node.sh

# coordinator (any one node). controls/ holds the control images workers fetch:
python3 fleet/coordinator.py --prompts prompts.txt --control-dir controls/ \
    --default-control plan.png --port 8765

# every worker, including the coordinator's own machine if it has a GPU:
source ~/sdvenv/bin/activate
python3 fleet/worker.py --coordinator http://10.10.10.1:8765
```

Each line of `prompts.txt` is one job and may end in ` | control=<file>` to
pick that job's control image from `--control-dir` (see `prompts.example.txt`).
Jobs without one use `--default-control`; a worker can also be given a local
`--control-image` as a last-resort fallback. Workers keep polling for new jobs
(`--exit-when-done` makes them exit once the queue is drained). A job that fails (e.g. out of VRAM) or goes quiet for 10 minutes is
retried up to 3 times, then listed under `failed` at `GET /status`. Run
`python3 -m pytest tests` to check the coordinator.

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

Written to be reproducible on anyone's set of low-VRAM laptops,
not specific to one machine or one project. If you build it out further
(the 2.5GbE switch version, the coarse rank0/rank1 split actually
generating images, or the full UNet pipeline split), a PR with what you
found — especially real `iperf3` numbers and per-step timings — is very
welcome.

### Running the coordinator as a service

`deploy/multitraktion-coordinator.service` is a systemd *user* unit (expects the
repo at `~/Projects/multitraktion`, with `prompts.txt` and `controls/plan.png`):

```
mkdir -p ~/.config/systemd/user && cp deploy/multitraktion-coordinator.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now multitraktion-coordinator
loginctl enable-linger $USER      # start at boot without a login
journalctl --user -u multitraktion-coordinator -f
```

## Tuning a node

The built-in defaults are the best config measured on the XPS 15 7590 (GTX 1650
Mobile, 4GB): fp32, sequential offload, `channels_last`, DPM++ at 12 steps, 512px
= ~28 s/img. A different GPU may want different settings, so tune it:

```
~/sdvenv/bin/python fleet/bench.py --tune-quality     # ~20-40 min, writes bench-best.json
cp bench-best.json worker-config.json                 # this node's override; the worker service picks it up
```

`bench-best.json` keeps full image quality (25 steps, default scheduler);
`bench-fast.json` (from `--tune-quality`) is the fastest config found. Look at
`bench-out/` before adopting either. For a one-off run:
`python3 fleet/worker.py --coordinator http://<coordinator>:8765 --config bench-fast.json`.

**fp16 is unusable on GTX 16xx (Turing TU117) with this stack:** the UNet and
ControlNet produce NaNs and every image is black. On Pascal (GTX 10xx) it is merely slow.
The benchmark rejects blank output and the worker refuses to deliver it, and the
default config is fp32 + sequential offload, which is correct on any GPU.
Measured on the XPS 15 7590 (GTX 1650 Mobile, 4GB): fp32 + sequential offload +
channels_last, 25 steps, 512px = 60.7 s/img; DPM++ at 12 steps = 28.3 s/img.

## LCM-LoRA preset (draft mode, not a replacement)

`fleet/config-lcm.json` renders in 4 steps with the LCM-LoRA merged into the UNet
and no unconditional pass (guidance 1.0), with the ControlNet weakened to 0.6 and
switched off after 60% of the steps. Measured on LAVE (GTX 1650): 7.5 s/img against
28-30 s/img for the default DPM++ 12-step config. Run it with
`worker.py --config fleet/config-lcm.json`. Needs `peft` (in requirements.txt).

**It is not quality-neutral.** On 8 real prompts the LCM output is flatter and more
saturated, and loses the weather, dusk lighting and railway context (tracks, steam
locomotive) that DPM++ keeps. Turning guidance up to 1.5 and the ControlNet down to
0.3 (`guidance`, `control_scale`, `control_end` in the config) recovers some subject
detail but costs the speed advantage (11 s/img, 2.5x) and does not recover the mood.
Use it for quick drafts and thumbnails; keep DPM++ 12 for final renders.
