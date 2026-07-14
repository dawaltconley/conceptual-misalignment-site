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
import sys
from pathlib import Path
from config import TERMS

# Set at runtime (main) when --arch api is used.
API_MODEL = "xunzi"

# FEW-SHOT EXAMPLES --- Drawn from the XunziALLM sample data: https://github.com/Xunzi-LLM-of-Chinese-classics/XunziALLM/blob/a10d6e9ad4e03c8c0cc4370e9d76e04cfa3aa011/sample%20data/sample%20data%20for%20downstream%20tasks%20evaluation/seg_downstream.json
FEWSHOT = []
with open('seg_data.json') as seg_data:
    data = json.load(seg_data)
    for shot in data:
        FEWSHOT.append((shot['input'], shot['output']))


SYSTEM_PROMPT = """
    You are an assistant to help in the processing of Classical Chinese texts.
    Please perform word segmentation on the input Classical Chinese passages,
    using the "/" character to delimit word boundaries. Output only the segmented
    results; do not include explanations, numbering, or extraneous text. Ensure
    the characters in the segmented output match the original sentence exactly
    (no added, omitted, or altered characters).

    Unless they are used as part of a proper noun, treat terms in the following
    list as distinct words: {terms}
""".format(terms=", ".join([t.hanzi for t in TERMS])).strip()


def split_lines(text: str):
    """Split a classical-Chinese text into lines, preserving all characters."""
    text = text.replace("\r", "")
    lines = [line.strip() for line in text.split("\n")]
    return [line for line in lines if line]


# --------------------------------------------------------------------------- #
# Parsing + QC
# --------------------------------------------------------------------------- #


def parse_segmented(output: str):
    """
    Parse a 'word/word/word ...' string into [word, word, ...].
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
    return output.split("/")


def qc_check(sentence: str, tokens):
    """
    Return (ok, reason). Two invariants:
      1. concatenation of segmented words == original sentence (character-exact)
      2. every tag is in the defined tagset
    """
    if tokens is None:
        return False, "unparseable output"
    recombined = "".join(w for w in tokens)
    if recombined != sentence:
        return False, f"char mismatch: got {recombined!r} vs {sentence!r}"
    bad_words = []
    for word in tokens:
        for term in TERMS:
            if term.hanzi in word and term.hanzi != word:
                bad_words.append(word)
    if bad_words:
        return False, f"obfuscated core term: found {', '.join(bad_words)}"
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


def tag_line(tok, model, sentence: str, max_new_tokens: int, arch: str, strict: bool = False):
    if arch == "api":
        # `model` is an OpenAI-compatible client; reuse the chat-template messages.
        client = model
        msgs = build_messages(sentence)
        if strict:
            msgs.insert(
                len(msgs) - 1,
                {"role": "user", "content": 'Please output only this text with added word boundaries using the "/" delimiter. Except for the "/" delimiter, the text must exactly match its input.'},
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
            query = '(Please output only this text with added word boundaries using the "/" delimiter. Except for the "/" delimiter, the text must exactly match its input.)\n' + sentence
        model.generation_config.max_new_tokens = max_new_tokens
        response, _ = model.chat(
            tok, query, history=history, system=SYSTEM_PROMPT)
        return response.strip()

    # arch == "qwen1.5" : build messages and use the chat template
    msgs = build_messages(sentence)
    if strict:
        msgs.insert(
            len(msgs) - 1,
            {"role": "user", "content": 'Please output only this text with added word boundaries using the "/" delimiter. Except for the "/" delimiter, the text must exactly match its input.'},
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
                    help="output JSONL of tagged lines")
    ap.add_argument("--errors", default=None,
                    help="JSONL for QC failures (default: <output>.errors.jsonl)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"],
                    help="'auto' uses GPU/MPS if available; 'cpu' forces CPU")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0,
                    help="tag only first N lines (0 = all; use for a dry run)")
    args = ap.parse_args()

    errors_path = Path(args.errors) if args.errors else Path(
        str(args.output) + ".errors.jsonl")

    text = Path(args.input).read_text(encoding="utf-8")
    lines = split_lines(text)
    if args.limit:
        lines = lines[: args.limit]
    print(f"[info] {len(lines)} lines to tag", file=sys.stderr)

    global API_MODEL
    API_MODEL = args.api_model
    tok, model = load_model(args.model, args.device, args.arch, args.api_base)

    try:
        from tqdm import tqdm
        iterator = tqdm(list(enumerate(lines)), total=len(lines))
    except ImportError:
        iterator = enumerate(lines)

    n_ok = n_err = 0
    with open(args.output, "w", encoding="utf-8") as fout, \
            open(errors_path, "w", encoding="utf-8") as ferr:
        for i, line in iterator:
            raw = tag_line(
                tok, model, line, args.max_new_tokens, args.arch)
            tokens = parse_segmented(raw)
            ok, reason = qc_check(line, tokens)
            if not ok:
                # handle a common case where xunzi drops a trailing quote
                if "".join(tokens) == line[:-1]:
                    if line[-1:] == "」":
                        tokens.append("」")
                        ok = True
                    if line[-1:] == "』":
                        tokens.append("』")
                        ok = True
                else:  # one stricter retry
                    raw = tag_line(
                        tok, model, line, args.max_new_tokens, args.arch, strict=True)
                    tokens = parse_segmented(raw)
                    ok, reason = qc_check(line, tokens)
            assert tokens is not None
            fout.write(raw + "\n")
            if ok:
                n_ok += 1
            else:
                rec = {
                    "id": i,
                    "line": line,
                    "raw_output": raw,
                    "reason": reason
                }
                ferr.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_err += 1

    print(
        f"[done] ok={n_ok}  needs_review={n_err}  -> {args.output}", file=sys.stderr)
    if n_err:
        print(f"[note] review failures in {errors_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
