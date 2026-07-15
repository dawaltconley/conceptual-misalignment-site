#!/usr/bin/env bash
set -euo pipefail

# Always operate from the scripts/ directory (where this file lives),
# regardless of where the script is invoked from.
cd "$(dirname "$0")"

echo ""
echo "==> Creating virtual env (.venv)..."
echo ""
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

echo ""
echo "==> Installing ROCm PyTorch..."
echo ""
pip install torch==2.9.1+rocm6.3 --index-url https://download.pytorch.org/whl/rocm6.3

echo ""
echo "==> Verifying GPU is visible to torch..."
echo ""
python -c "import torch; assert torch.cuda.is_available(), 'ROCm GPU not visible to torch'; print('GPU:', torch.cuda.get_device_name(0))"

# ---------------------------------------------------------------------------
# 2. Everything else from PyPI. numpy is pinned to 2.4.4 in requirements.txt
#    so this step cannot silently downgrade it and break scipy/sklearn/cltk.
# ---------------------------------------------------------------------------
echo ""
echo "==> Installing project requirements..."
echo ""
pip install -r requirements.txt

# spaCy English pipeline loaded by nlp/english.py (spacy.load("en_core_web_sm"))
echo ""
echo "==> Downloading spaCy en_core_web_sm model..."
echo ""
python -m spacy download en_core_web_sm

# Warm up the SuPar-Kanbun classical-Chinese model (nlp/chinese.py) so the first
# main.py run doesn't pause to download the transformer weights.
echo ""
echo "==> Downloading SuPar-Kanbun classical-Chinese model..."
echo ""
python -c "import suparkanbun; suparkanbun.load()"

echo ""
echo "==> Sanity check..."
echo ""
python -m pip check

echo ""
echo "==> Done. Activate with:  source scripts/.venv/bin/activate"
