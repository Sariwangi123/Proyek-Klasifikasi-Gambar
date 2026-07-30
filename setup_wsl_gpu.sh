#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-wsl-gpu.txt
python -m pip install tensorflowjs==4.22.0 --no-deps
python check_gpu.py

echo "Setup WSL GPU selesai. Jalankan: source .venv-wsl/bin/activate && python train_local_gpu.py"
