#!/usr/bin/env bash
set -euo pipefail

# Provision the scripts/ environment with uv.
# Dependencies live in pyproject.toml; the exact resolution is locked in uv.lock.
# The ROCm PyTorch build is pulled from the pytorch-rocm index declared in
# pyproject.toml — no manual index ordering needed.
#
#   Install uv (once):  curl -LsSf https://astral.sh/uv/install.sh | sh
#   Run a script:       uv run python xunzi_seg.py ...

cd "$(dirname "$0")"

echo "==> uv sync..."
uv sync

echo "==> Verifying ROCm GPU is visible to torch..."
uv run python -c "import torch; assert torch.cuda.is_available(), 'ROCm GPU not visible to torch'; print('GPU:', torch.cuda.get_device_name(0))"

echo "==> Done. Run scripts with:  uv run python <script>.py"
