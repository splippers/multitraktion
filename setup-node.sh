#!/usr/bin/env bash
# Builds the worker environment on a fresh Ubuntu node and prints the
# versions to compare against the reference machine. Run once per node.
set -euo pipefail

command -v nvidia-smi >/dev/null || {
  echo "No NVIDIA driver found. Install one first (ubuntu-drivers install), reboot, rerun." >&2
  exit 1
}
sudo apt-get install -y python3-venv python3-pip iperf3 bolt

python3 -m venv ~/sdvenv
source ~/sdvenv/bin/activate
pip install --upgrade pip
# cu121 wheels bundle their own CUDA runtime; only the driver must be recent enough
pip install -r "$(dirname "$0")/fleet/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu121

echo "--- node fingerprint (must match on every node) ---"
lsb_release -ds
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3 --version
pip freeze | grep -iE '^(torch|diffusers|transformers|accelerate)=='
python3 -c "import torch; print('cuda available:', torch.cuda.is_available())"
