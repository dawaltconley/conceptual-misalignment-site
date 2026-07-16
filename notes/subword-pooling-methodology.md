# Subword & occurrence pooling: what the pipeline does and why

How a term's vector is built in `scripts/embed/`, and the literature supporting
each step. Relevant to defending the method in the dissertation.

## What the code does

Building a term vector happens in **two distinct pooling stages**:

### Stage 1 — mean over subword/character tokens *within* one occurrence

In `model.py` (`_embed_batch`), for each occurrence of a word we gather the
final-layer hidden states of every model token overlapping the word's character
span and average them:

```python
vec = hidden[i, tok_idx].mean(axis=0)   # mean over the word's tokens, this occurrence
```

So if a word is split into several subword/character tokens *within that single
occurrence*, those pieces are mean-pooled into one occurrence vector.

### Stage 2 — max-pool *across* occurrences

In `vectors.py`, all the occurrence vectors for a word are stacked and
**element-wise max-pooled** into one vector per word. This second stage is the
Wu & Wang (2025) choice the project follows (keeps the dominant sense, reduces
subword-fragmentation noise) — it is *not* from the general subword literature.

## Literature support

### Stage 1 (mean over subwords) — a standard, validated baseline

- **Bommasani, Davis & Cardie (2020), "Interpreting Pretrained Contextualized
  Representations via Reductions to Static Embeddings," ACL 2020.** The most
  on-point single source: it forms a word representation by pooling its subword
  tokens, *then* pools across many contexts to get a static vector — i.e. it
  covers **both** of our stages. <https://aclanthology.org/2020.acl-main.431/>
- **Vulić, Ponti, Litschko, Glavaš & Korhonen (2020), "Probing Pretrained
  Language Models for Lexical Semantics," EMNLP 2020.** Builds type-level word
  vectors from BERT/RoBERTa by averaging subwords and averaging over contexts.
  Also reports that ~10 contexts is enough to stabilize a type vector — our
  target counts (仁=151, 義=105, 禮=66, 智=30) are comfortably above that.
  <https://aclanthology.org/2020.emnlp-main.586/>
- **Ács, Kádár & Kornai (2021), "Subword Pooling Makes a Difference," EACL
  2021.** Studies the choice head-on (first / last / **mean** / attention / LSTM
  over subwords). Cite honestly: mean is a fine baseline but *not always
  optimal* — "choose the first subword" is worst, and attention/LSTM pooling
  often wins on morphology / tagging tasks.
  <https://aclanthology.org/2021.eacl-main.194/>

### Stage 2 (max across occurrences)

- **Wu & Wang (2025), npj Heritage Science, DOI 10.1038/s40494-025-01893-7** —
  the source method being adapted; max-pooling across occurrences comes from here.

## Why the subword nuance barely matters for our targets

Ács's "it makes a difference" warning applies mostly to heavily-subworded
WordPiece / BPE **multilingual** models. **GujiRoBERTa tokenizes Chinese one
token per character**, verified three ways from the shipped tokenizer (the HF
model card is an auto-generated stub and documents none of this — the tokenizer
artifacts are the authoritative source). Note the precise claim is about
*tokenization granularity*: the model is pretrained on running text and still
models context across characters via self-attention; each hanzi is simply one
input token → one hidden vector.

1. **Config forces it.** `BertTokenizer` with `tokenize_chinese_chars: true`
   (fast tokenizer: `BertNormalizer` `handle_chinese_chars: true`). That flag
   inserts whitespace around every CJK codepoint before WordPiece, pre-splitting
   each character into its own unit regardless of vocab.
2. **Vocab makes multi-char CJK tokens impossible.** Of 34,709 entries:
   **28,223 single CJK characters, 0 multi-character CJK tokens** (the rest are
   9,614 `##` continuation pieces + 6,382 Latin/digit/punct entries). There is
   no 孟子 / 天下 token to assign, even in principle.
3. **Behavior confirms it:** 孟子→孟,子 · 天下→天,下 · 梁惠王→梁,惠,王 ·
   仁義→仁,義 · 孔子曰仁→孔,子,曰,仁. Every hanzi is its own token.

Reproduce with `AutoTokenizer.from_pretrained("hsc748NLP/GujiRoBERTa_fan")` and
inspect `get_vocab()` / tokenize a few multi-char words. Caveat: a rare
out-of-vocab hanzi becomes a single `[UNK]` token — still one token per
character, just occasionally as UNK (worth spot-checking the targets + frequent
neighbors don't hit UNK).

Critically, the four targets **仁 義 禮 智 are single characters** → each
occurrence is exactly one token, so the Stage-1 mean is a **no-op for the
targets themselves**. The subword mean only ever activates for multi-character
*vocabulary neighbors* (e.g. 孟子, 天下), and even there it is char-level
averaging, not aggressive subword fragmentation. So our exposure to Ács's caveat
is low; Bommasani / Vulić remain the defense that mean-pooling subwords into a
token vector is standard and validated if a reviewer pushes.
