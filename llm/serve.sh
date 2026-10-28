#!/bin/bash
# Serves the Qwen3-Coder-30B-A3B MoE model with tool calling on a 4 GB GPU.
# Attention + KV cache live on the GPU, most expert weights stay in system RAM
# (--n-cpu-moe 44 leaves 4 layers' experts on the 4 GB GPU; 43 does not fit).
# Measured on an XPS 15 7590 (GTX 1650, 6C/12T i7-9750H):
# ~18 tok/s generation, ~90 tok/s prompt processing at 16k context, vs 3.7 / 52
# with Ollama's default layer split. Needs the GPU to itself (close BeamNG).
# Build llama.cpp with: cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
#   -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-13
# (CUDA 12.4 rejects gcc 15.) Use -t = physical cores, not threads.
set -euo pipefail
cd "${LLAMA_CPP:-$HOME/llama.cpp}"
MODEL=${MODEL:?set MODEL to the qwen3-coder 30B GGUF (e.g. the Ollama blob)}
# With less than ~3 GB free VRAM, set GGML_CUDA_DISABLE_GRAPHS=1 and --n-cpu-moe 48.
exec ./build/bin/llama-server -m "$MODEL" -ngl 99 --n-cpu-moe "${NCPUMOE:-44}" -fa on -t 6 -ub 512 \
  -c "${CTX:-16384}" -ctk q8_0 -ctv q8_0 --jinja --host 127.0.0.1 --port "${PORT:-8090}"
