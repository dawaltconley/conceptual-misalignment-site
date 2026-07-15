#!/usr/bin/env bash
set -euo pipefail

# Always operate from the scripts/ directory (where this file lives),
# regardless of where the script is invoked from.
cd "$(dirname "$0")"

echo "==> Creating virtual env (.venv)..."
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

echo "==> Installing ROCm PyTorch..."
pip install torch==2.9.1+rocm6.3 --index-url https://download.pytorch.org/whl/rocm6.3

echo "==> Verifying GPU is visible to torch..."
python -c "import torch; assert torch.cuda.is_available(), 'ROCm GPU not visible to torch'; print('GPU:', torch.cuda.get_device_name(0))"

# ---------------------------------------------------------------------------
# 2. Everything else from PyPI. numpy is pinned to 2.4.4 in requirements.txt
#    so this step cannot silently downgrade it and break scipy/sklearn/cltk.
# ---------------------------------------------------------------------------
echo "==> Installing project requirements..."
pip install -r requirements.txt

# spaCy English pipeline loaded by nlp/english.py (spacy.load("en_core_web_sm"))
echo "==> Downloading spaCy en_core_web_sm model..."
python -m spacy download en_core_web_sm

echo "==> Sanity check..."
python -m pip check

echo ""
echo "==> Done. Activate with:  source scripts/.venv/bin/activate"
