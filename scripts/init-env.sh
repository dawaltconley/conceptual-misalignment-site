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
# 2. Everything else from PyPI, via the editable install of the project defined
#    in ../pyproject.toml. numpy is pinned to >=2.4.4 there so this step cannot
#    silently downgrade it and break scipy/sklearn/cltk. torch (installed above
#    from the rocm index) already satisfies its pin, so it is not re-fetched.
# ---------------------------------------------------------------------------
echo ""
echo "==> Installing the project (editable) + dependencies..."
echo ""
pip install -e "..[dev]"

# spaCy English pipeline loaded by nlp/english.py (spacy.load("en_core_web_sm"))
echo ""
echo "==> Downloading spaCy en_core_web_sm model..."
echo ""
python -m spacy download en_core_web_sm

echo ""
echo "==> Sanity check..."
echo ""
python -m pip check

echo ""
echo "==> Done. Activate with:  source scripts/.venv/bin/activate"
