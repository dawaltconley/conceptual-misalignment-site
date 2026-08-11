# GujiRoBERTa model loading & the transformers LOAD REPORT

Notes on how the embedding pipeline loads the model, whether it reuses the local
HuggingFace cache, and how to read the `BertModel LOAD REPORT` that transformers
prints on load.

## 1. Does the pipeline reuse the locally-downloaded model?

**Yes, automatically.** In `scripts/embed/model.py` we call:

```python
AutoTokenizer.from_pretrained(model_name, use_fast=True)
AutoModel.from_pretrained(model_name)
```

`from_pretrained` treats `model_name` (e.g. `hsc748NLP/GujiRoBERTa_jian_fan`) as a
**repo id** and resolves it against the HuggingFace cache at
`~/.cache/huggingface/hub/`. If the snapshot is already present, it loads the
weights **from disk** — no re-download. There is nothing to point at manually; the
cache is the default source.

The `unauthenticated requests to the HF Hub` warning you may see is **not** a
re-download — it's a lightweight metadata check to see whether a newer revision
exists on the Hub. To skip even that and go fully cache-only:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

Passing a local filesystem path instead of the repo id also works, but is
unnecessary and less portable — prefer the repo-id + cache mechanism.

### Which variant? `_fan` vs `_jian_fan`

Two variants from the same authors are cached locally, **same architecture**
(BERT, `hidden_size` 768, `vocab_size` 34709):

| Repo | Trained on | Note |
|------|-----------|------|
| `hsc748NLP/GujiRoBERTa_jian_fan` | 简+繁 (simplified **and** traditional) | what the plan / first full run used |
| `hsc748NLP/GujiRoBERTa_fan` | 繁 (traditional only) | arguably a cleaner match for the traditional Mengzi corpus |

Because the Mengzi corpus is **traditional**, the traditional-only `_fan` model is
a defensible choice. Whichever you standardize on, keep `DEFAULT_MODEL` and the
artifacts in `analysis/task1/` consistent: **re-run the pipeline after changing
the model** so the saved vectors match the model that produced them.

## 2. Reading the `BertModel LOAD REPORT`

```
[transformers] BertModel LOAD REPORT from: hsc748NLP/GujiRoBERTa_jian_fan
Key                                        | Status     |
cls.predictions.decoder.bias               | UNEXPECTED |
cls.predictions.transform.dense.weight     | UNEXPECTED |
cls.predictions.transform.dense.bias       | UNEXPECTED |
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |
cls.predictions.decoder.weight             | UNEXPECTED |
cls.predictions.bias                       | UNEXPECTED |
pooler.dense.weight                        | MISSING    |
pooler.dense.bias                          | MISSING    |
```

**Both sections are benign for our use — the encoder itself loaded completely.**

The reason there is any mismatch at all:

- The checkpoint on the Hub is a **masked-LM** checkpoint = encoder body **+** an
  MLM prediction head.
- We instantiate `AutoModel`, which for `model_type: bert` resolves to
  **`BertModel`**: the bare encoder, **no task head**. Hence "BertModel LOAD
  REPORT".

### UNEXPECTED — `cls.predictions.*` (7 keys)

These are the **MLM head**: the decoder projecting hidden states to vocabulary
logits, plus its transform layer (dense + LayerNorm) and biases. `BertModel` has
no such head, so these checkpoint weights are **dropped**. We only read
`last_hidden_state`, so discarding the MLM head is exactly what we want.

### MISSING — `pooler.dense.*` (2 keys)

`BertModel` includes an optional **pooler** — a `dense` + `tanh` over the `[CLS]`
token, used for sentence-level classification. MLM pretraining never trains a
pooler, so the checkpoint has no pooler weights and they are **randomly
initialized**. This matters only if you read `pooler_output`. **We don't** — we
mean-pool token vectors from `last_hidden_state` — so the random pooler is never
invoked and cannot affect the embeddings.

### Why this is safe

The report lists **only** the discarded MLM head and the unused pooler. Every
encoder weight (`embeddings.*`, all `encoder.layer.*`) matched silently. If any of
*those* were UNEXPECTED / MISSING, that would be a real problem. The
"Consider training on your downstream task" line is generic HuggingFace
boilerplate aimed at fine-tuners; it does not apply to feature extraction.

**Net: ignore both sections. The vectors come entirely from correctly-loaded
encoder weights.**
