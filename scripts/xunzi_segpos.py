"""
xunzi_segpos.py
Programmatic word segmentation + POS tagging of classical Chinese with the
Xunzi-Qwen1.5-7B_chat model.

Pipeline:
  raw text  ->  sentence split  ->  per-sentence LLM tagging  ->  QC  ->  JSONL

Design choices worth knowing:
  * The output format and the tagset are pinned by FEW-SHOT EXAMPLES, not by
    trusting the model's defaults. This makes results reproducible regardless of
    how the model was fine-tuned. ==> You must replace the FEWSHOT examples below
    with 2-3 sentences you have hand-verified to YOUR segmentation/POS standard.
  * Decoding is greedy (do_sample=False) for determinism.
  * Every tagged sentence is QC-checked: the concatenation of the segmented
    tokens must equal the original sentence exactly (no dropped/added/altered
    characters). Failures are retried once, then logged for manual review rather
    than silently written.

Usage:
  python xunzi_segpos.py --input mengzi.txt --output mengzi.segpos.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# POS tagset. A compact, classical-Chinese-appropriate set. ADJUST to match the
# reference standard you will validate against; the distinction of modal/final
# particles (y) matters a great deal for 文言文.
# --------------------------------------------------------------------------- #
TAGSET = {
    "n":  "名词 noun",
    "nr": "人名 person name",
    "ns": "地名 place name",
    "t":  "时间词 time word",
    "v":  "动词 verb",
    "a":  "形容词 adjective",
    "d":  "副词 adverb",
    "r":  "代词 pronoun",
    "m":  "数词 numeral",
    "q":  "量词 classifier",
    "p":  "介词 preposition",
    "c":  "连词 conjunction",
    "u":  "助词 structural particle (之/所/者 ...)",
    "y":  "语气词 modal/final particle (也/矣/乎/哉 ...)",
    "w":  "标点 punctuation",
}
VALID_TAGS = set(TAGSET)

# Set at runtime (main) when --arch api is used.
API_MODEL = "xunzi"

# --------------------------------------------------------------------------- #
# FEW-SHOT EXAMPLES  ---  PLACEHOLDERS. Replace with your own gold-verified
# annotations. These define the output format ("word/TAG word/TAG ...") and the
# tagset by demonstration. Two or three short, function-word-rich sentences from
# the Mengzi work best. The segmentation/tags below are a plausible first pass
# and are NOT authoritative -- you are the philologist; make them match your
# standard before trusting bulk output.
# --------------------------------------------------------------------------- #
FEWSHOT = [
    (
        "孟子見梁惠王。",
        "孟子/nr 見/v 梁惠王/nr 。/w",
    ),
    (
        "王何必曰利？",
        "王/n 何/r 必/d 曰/v 利/n ？/w",
    ),
    (
        "亦有仁義而已矣。",
        "亦/d 有/v 仁義/n 而已/y 矣/y 。/w",
    ),
]

SYSTEM_PROMPT = (
    "你是古汉语文本处理助手。请对输入的古文句子进行分词和词性标注。"
    "输出格式为“词/词性 词/词性 …”，词与词之间用一个空格分隔，"
    "每个词后用斜杠加词性标记。词性标记集为："
    + "，".join(f"{k}={v}" for k, v in TAGSET.items())
    + "。只输出标注结果，不要输出任何解释、编号或多余文字，"
    "保证标注结果中的字与原句完全一致（不增字、不减字、不改字）。"
)

# --------------------------------------------------------------------------- #
# Sentence splitting
# --------------------------------------------------------------------------- #
# Split after 。！？；, keeping the delimiter, and pulling any trailing closing
# quotes/brackets back onto the sentence they close.
_SENT_RE = re.compile(r"[^。！？；]*[。！？；]+[」』）】》\)]*")


def split_sentences(text: str):
    """Split a classical-Chinese text into sentences, preserving all characters."""
    text = text.replace("\r", "")
    sentences = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        pos = 0
        for m in _SENT_RE.finditer(line):
            s = m.group().strip()
            if s:
                sentences.append(s)
            pos = m.end()
        # trailing text with no terminal punctuation
        tail = line[pos:].strip()
        if tail:
            sentences.append(tail)
    return sentences


# --------------------------------------------------------------------------- #
# Parsing + QC
# --------------------------------------------------------------------------- #
def parse_tagged(output: str):
    """
    Parse a 'word/TAG word/TAG ...' string into [(word, tag), ...].
    Tolerates stray whitespace/newlines. Splits each token on its LAST slash so
    that a word containing '/' would still parse (rare in this domain).
    Returns None if any token is malformed.
    """
    output = output.strip()
    # If the model wrapped the answer in extra lines, keep the line that looks
    # most like a tagged sequence (contains '/').
    candidate_lines = [ln for ln in output.splitlines() if "/" in ln]
    if candidate_lines:
        output = max(candidate_lines, key=len)
    tokens = []
    for chunk in output.split():
        if "/" not in chunk:
            return None
        word, _, tag = chunk.rpartition("/")
        if not word or not tag:
            return None
        tokens.append((word, tag))
    return tokens or None


def qc_check(sentence: str, tokens):
    """
    Return (ok, reason). Two invariants:
      1. concatenation of segmented words == original sentence (character-exact)
      2. every tag is in the defined tagset
    """
    if tokens is None:
        return False, "unparseable output"
    recombined = "".join(w for w, _ in tokens)
    if recombined != sentence:
        return False, f"char mismatch: got {recombined!r} vs {sentence!r}"
    bad = sorted({t for _, t in tokens if t not in VALID_TAGS})
    if bad:
        return False, f"unknown tags: {bad}"
    return True, "ok"


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_messages(sentence: str):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for src, tagged in FEWSHOT:
        msgs.append({"role": "user", "content": src})
        msgs.append({"role": "assistant", "content": tagged})
    msgs.append({"role": "user", "content": sentence})
    return msgs


def load_model(model_id: str, device: str, arch: str, api_base: str | None = None):
    """Lazy import so the parsing/QC logic can be used without torch installed."""
    if arch == "api":
        # No local model: return an OpenAI-compatible client in the `model` slot.
        from openai import OpenAI
        client = OpenAI(base_url=api_base, api_key="not-needed")
        return None, client

    import torch

    if arch == "qwen1":
        # Original Qwen (v1) ships custom modeling code, so trust_remote_code is
        # required. It also needs:  pip install tiktoken transformers_stream_generator einops
        # and a transformers version compatible with the bundled code -- the v1
        # code is happiest on transformers 4.32-4.33; newer versions can break it.
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True,
            device_map="auto" if device != "cpu" else None,
        ).eval()
        if device == "cpu":
            # transformers' _Wrapped stub for Module.to mangles the signature;
            # the runtime call is correct.
            model = model.float().to("cpu")  # pyright: ignore
        try:
            model.generation_config = GenerationConfig.from_pretrained(
                model_id, trust_remote_code=True)
        except Exception:
            pass
        model.generation_config.do_sample = False   # deterministic
        return tok, model

    # arch == "qwen1.5" : native transformers, no remote code
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto" if device != "cpu" else None
    ).eval()
    if device == "cpu":
        model = model.to("cpu")  # pyright: ignore[reportArgumentType]
    return tok, model


def tag_sentence(tok, model, sentence: str, max_new_tokens: int, arch: str, strict: bool = False):
    if arch == "api":
        # `model` is an OpenAI-compatible client; reuse the chat-template messages.
        client = model
        msgs = build_messages(sentence)
        if strict:
            msgs.insert(
                len(msgs) - 1,
                {"role": "user", "content": "请严格只输出“词/词性”序列，字数须与原句完全相同。"},
            )
        resp = client.chat.completions.create(
            model=API_MODEL, messages=msgs, temperature=0.0, max_tokens=max_new_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    import torch

    if arch == "qwen1":
        # Original Qwen exposes a .chat() helper; few-shot goes in `history`
        # as a list of [user, assistant] pairs, and the system prompt via system=.
        history = [[src, tagged] for src, tagged in FEWSHOT]
        query = sentence
        if strict:
            query = "（严格只输出“词/词性”序列，字数须与原句完全相同）\n" + sentence
        model.generation_config.max_new_tokens = max_new_tokens
        response, _ = model.chat(
            tok, query, history=history, system=SYSTEM_PROMPT)
        return response.strip()

    # arch == "qwen1.5" : build messages and use the chat template
    msgs = build_messages(sentence)
    if strict:
        msgs.insert(
            len(msgs) - 1,
            {"role": "user", "content": "请严格只输出“词/词性”序列，字数须与原句完全相同。"},
        )
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            repetition_penalty=1.0, pad_token_id=tok.eos_token_id,
        )
    gen = out[0][inputs.input_ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Xunzi seg+POS for classical Chinese")
    ap.add_argument("--model", default="./Xunzi-Qwen-Chat",
                    help="local path to model dir (from `ms cache scan`), or an HF id "
                         "such as ccwu0918/XunziALLM")
    ap.add_argument("--arch", default="qwen1", choices=["qwen1", "qwen1.5", "api"],
                    help="qwen1 = original Qwen (Xunzi-Qwen-Chat: needs trust_remote_code + .chat); "
                         "qwen1.5 = Qwen1.5/2 (Xunzi-Qwen1.5-7B_chat: uses the chat template); "
                         "api = OpenAI-compatible server (llama.cpp/Ollama/hosted) -- no local load")
    ap.add_argument("--api-base", default="http://127.0.0.1:8080/v1",
                    help="OpenAI-compatible base URL (used only with --arch api)")
    ap.add_argument("--api-model", default="xunzi",
                    help="model name to send to the API (llama.cpp ignores it; Ollama needs the tag)")
    ap.add_argument("--input", required=True,
                    help="UTF-8 text file (e.g. mengzi.txt)")
    ap.add_argument("--output", required=True,
                    help="output JSONL of tagged sentences")
    ap.add_argument("--errors", default=None,
                    help="JSONL for QC failures (default: <output>.errors.jsonl)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"],
                    help="'auto' uses GPU/MPS if available; 'cpu' forces CPU")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0,
                    help="tag only first N sentences (0 = all; use for a dry run)")
    args = ap.parse_args()

    errors_path = Path(args.errors) if args.errors else Path(
        str(args.output) + ".errors.jsonl")

    text = Path(args.input).read_text(encoding="utf-8")
    sentences = split_sentences(text)
    if args.limit:
        sentences = sentences[: args.limit]
    print(f"[info] {len(sentences)} sentences to tag", file=sys.stderr)

    global API_MODEL
    API_MODEL = args.api_model
    tok, model = load_model(args.model, args.device, args.arch, args.api_base)

    try:
        from tqdm import tqdm
        iterator = tqdm(list(enumerate(sentences)), total=len(sentences))
    except ImportError:
        iterator = enumerate(sentences)

    n_ok = n_err = 0
    with open(args.output, "w", encoding="utf-8") as fout, \
            open(errors_path, "w", encoding="utf-8") as ferr:
        for i, sent in iterator:
            raw = tag_sentence(
                tok, model, sent, args.max_new_tokens, args.arch)
            tokens = parse_tagged(raw)
            ok, reason = qc_check(sent, tokens)
            if not ok:  # one stricter retry
                raw = tag_sentence(
                    tok, model, sent, args.max_new_tokens, args.arch, strict=True)
                tokens = parse_tagged(raw)
                ok, reason = qc_check(sent, tokens)
            rec = {"id": i, "sentence": sent}
            assert tokens is not None
            if ok:
                rec["tokens"] = [{"word": w, "pos": t} for w, t in tokens]
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
            else:
                rec["raw_output"] = raw
                rec["reason"] = reason
                ferr.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_err += 1

    print(
        f"[done] ok={n_ok}  needs_review={n_err}  -> {args.output}", file=sys.stderr)
    if n_err:
        print(f"[note] review failures in {errors_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
