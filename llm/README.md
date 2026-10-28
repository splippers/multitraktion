# Local tool-calling LLM

Measured on LAVE (XPS 15 7590, GTX 1650 4 GB, 30 GB RAM), 2026-10-28. Rows 1-2
and the RPC row ran with ~1.8 GB VRAM free (BeamNG held 2.2 GB; the desktop
itself runs on the Intel GPU and uses ~0); the last llama.cpp row had the GPU
to itself.

| Setup | Generation | Prompt processing | Tool calls |
|---|---|---|---|
| Ollama qwen3-coder:30b (default split) | 3.7 tok/s | 52 tok/s | yes |
| llama.cpp, experts on CPU (`serve.sh`) | 16-18 tok/s | 60-75 tok/s | yes (verified, `--jinja`) |
| **llama.cpp, GPU free, 4 layers' experts on GPU, 16k ctx (`serve.sh`)** | 18.4 tok/s (12-15 on short agent replies) | 91 tok/s (78 in the agent test) | yes |
| Ollama qwen3:4b | 5.5-6.6 tok/s | 139 tok/s | yes |
| Ollama qwen2.5-coder:7b | 5.5 tok/s | 96 tok/s | no - emits JSON as text |
| Ollama qwen2.5-coder:14b | 1.4 tok/s | 45 tok/s | not tested |
| llama.cpp RPC, half the experts on a loopback RPC server | 6.2 tok/s | 29 tok/s | - |

Notes
- Prompt processing, not generation, is the agent bottleneck. Follow-up turns
  reuse the prompt cache (2431 of 2471 tokens cached in the agent test).
- `-ub 512` and larger OOM on this GPU; `-t 6` beats 8 and 12.
- The RPC row is confounded (client and server shared one CPU), but llama.cpp
  also runs the local and RPC portions sequentially, so splitting experts
  across two machines this way cannot double memory bandwidth.

## Service

`deploy/multitraktion-llm.service` is a user unit (installed like the
coordinator's). It serves an OpenAI-compatible API with tool calling on
`127.0.0.1:8090` only; it holds ~3.4 GB VRAM and ~14 GB RAM, so
`systemctl --user stop multitraktion-llm` before running the image worker or a game.
`MODEL` points at Ollama's blob for qwen3-coder:30b - update it if Ollama prunes it.
