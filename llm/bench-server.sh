#!/bin/bash
# usage: srvbench.sh <label> <server args...>   -> prints prefill/gen t/s for a fixed request
cd "${LLAMA_CPP:-$HOME/llama.cpp}"; export GGML_CUDA_DISABLE_GRAPHS=1
M=/usr/share/ollama/.ollama/models/blobs/sha256-1194192cf2a187eb02722edcc3f77b11d21f537048ce04b67ccf8ba78863006a
label=$1; shift
pkill -x llama-server; sleep 1
setsid nohup ./build/bin/llama-server -m $M -ngl 99 -fa on -t 6 -ub 256 -c 4096 -ctk q8_0 -ctv q8_0 --port 8090 --host 127.0.0.1 --no-warmup "$@" > srv.log 2>&1 < /dev/null &
for i in $(seq 1 120); do curl -sf localhost:8090/health >/dev/null && break; grep -qE "loading error|Aborted|CUDA error" srv.log && { echo "$label: LOAD FAIL: $(grep -E ' E |error' srv.log | head -2 | cut -c1-160)"; pkill -x llama-server; exit 1; }; sleep 2; done
python3 - "$label" <<'PY'
import json,sys,urllib.request
p="".join(f"def f{i}(x):\n    return x * {i} + 1\n\n" for i in range(60))
b={"prompt":p+"# next:","n_predict":48,"temperature":0,"cache_prompt":False}
r=urllib.request.Request("http://127.0.0.1:8090/completion",json.dumps(b).encode(),{"Content-Type":"application/json"})
t=json.loads(urllib.request.urlopen(r,timeout=900).read())["timings"]
print(f"{sys.argv[1]}: prefill {t['prompt_n']} tok @ {t['prompt_per_second']:.0f} t/s | gen {t['predicted_n']} tok @ {t['predicted_per_second']:.1f} t/s")
PY
pkill -x llama-server
