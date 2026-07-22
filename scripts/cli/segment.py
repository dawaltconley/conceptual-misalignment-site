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
from typing import Literal
from dataclasses import dataclass, asdict
from pathlib import Path
from difflib import ndiff
from config import TERMS
from corpus.mengzi import Chapter, fetch_mengzi_full

import segmentation.seg
import segmentation.segpos
from segmentation.utils import ARCH, assert_arch

# Set at runtime (main) when --arch api is used.
API_MODEL = "xunzi"


# Set at runtime (main); when True, api requests are constrained by a per-sentence
# GBNF grammar so the model can only emit the exact input characters.
USE_GRAMMAR = False

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


class Word:
    def __init__(self, word: str, pos: str | None = None):
        self.word = word
        self.pos = pos

    def serialize(self):
        if self.pos:
            return {"word": self.word, "pos": self.pos}
        return self.word


class SegPos:
    arch: ARCH

    def __init__(self, process: Literal['seg', 'segpos'], *, system_prompt: str, fewshot: list[tuple[str, str]], arch: ARCH, core_terms: list[str] = [], max_new_tokens: int = 2048):
        self.process: Literal['seg', 'segpos'] = process
        self.system_prompt = system_prompt
        if core_terms:
            self.system_prompt += "\n\n"
            "Unless they are used as part of a proper noun, treat terms in the following"
            "list as distinct words: {terms}".format(
                terms=", ".join(core_terms))
        self.fewshot = fewshot
        self.core_terms = core_terms
        self.arch = arch
        self.max_new_tokens = max_new_tokens

    def parse(self, output: str) -> list[Word] | None:
        if self.process == "seg":
            # words = [Word(w) for w in segmentation.seg.parse(output)]
            parsed = segmentation.seg.parse(output)
            if not parsed:
                return None
            return [Word(w) for w in parsed]
        elif self.process == "segpos":
            parsed = segmentation.segpos.parse(output)
            if not parsed:
                return None
            return [Word(w, pos) for w, pos in parsed]

    def qc_check(self, sentence: str, tokens: list[Word] | None) -> tuple[bool, str]:
        """
        Return (ok, reason). Two invariants:
          1. concatenation of segmented words == original sentence (character-exact)
          2. every tag is in the defined tagset
        """
        if tokens is None:
            return False, "unparseable output"
        error: list[str] = []
        recombined = "".join(t.word for t in tokens)
        if recombined != sentence:
            error.append(f"char mismatch: got {recombined!r} vs {sentence!r}")
        if self.process == "segpos":
            bad_tags = []
            for t in tokens:
                if not (t.pos and t.pos in segmentation.segpos.VALID_TAGS):
                    bad_tags.append(t.pos)
            if bad_tags:
                error.append(f"unknown tags: {sorted(bad_tags)}")
        bad_words = []
        for t in tokens:
            for term in self.core_terms:
                if term in t.word and term != t.word:
                    bad_words.append(t.word)
                    break
        if bad_words:
            error.append(f"obfuscated core term: found {', '.join(bad_words)}")
        if error:
            return False, "; ".join(error)
        return True, "ok"

    def validate_tokens(self, tokens: list[Word]) -> None:
        for t in tokens:
            if self.process == "seg" and t.pos != None:
                raise ValueError(
                    "Bad token, found pos during segmentation: " + t.pos)
            elif self.process == "segpos" and t.pos == None:
                raise ValueError("Bad token, no pos during segpos: " + t.word)

    def build_messages(self, sentence: str):
        msgs = [{"role": "system", "content": self.system_prompt}]
        for src, tagged in self.fewshot:
            msgs.append({"role": "user", "content": src})
            msgs.append({"role": "assistant", "content": tagged})
        msgs.append({"role": "user", "content": sentence})
        return msgs

    def tag_sentence(self, *, tok, model, sentence: str, strict: bool = False):
        strict_prompt: str
        if self.process == "seg":
            strict_prompt = segmentation.seg.get_strict_prompt(self.arch)
        elif self.process == "segpos":
            strict_prompt = segmentation.segpos.get_strict_prompt(self.arch)
        else:
            raise ValueError(f"Bad process value: {self.process!r}")

        if self.arch == "api":
            # `model` is an OpenAI-compatible client; reuse the chat-template messages.
            client = model
            msgs = self.build_messages(sentence)
            if strict:
                msgs.insert(
                    len(msgs) - 1,
                    {"role": "user", "content": strict_prompt},
                )
            grammar = None
            if USE_GRAMMAR:
                if self.process == "seg":
                    grammar = build_gbnf(sentence, word_break="/")
                elif self.process == "segpos":
                    tags = sorted(segmentation.segpos.VALID_TAGS)
                    grammar = build_gbnf(sentence, tags, word_break=" ")
            resp = client.chat.completions.create(
                model=API_MODEL, messages=msgs, temperature=0.0, max_tokens=self.max_new_tokens,
                extra_body={"grammar": grammar} if grammar else {},
            )
            return (resp.choices[0].message.content or "").strip()

        import torch

        if self.arch == "qwen1":
            # Original Qwen exposes a .chat() helper; few-shot goes in `history`
            # as a list of [user, assistant] pairs, and the system prompt via system=.
            history = [[src, tagged] for src, tagged in self.fewshot]
            query = sentence
            if strict:
                query = strict_prompt + sentence
            model.generation_config.max_new_tokens = self.max_new_tokens
            response, _ = model.chat(
                tok, query, history=history, system=self.system_prompt)
            return response.strip()

        # arch == "qwen1.5" : build messages and use the chat template
        msgs = self.build_messages(sentence)
        if strict:
            msgs.insert(
                len(msgs) - 1,
                {"role": "user", "content": strict_prompt},
            )
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                repetition_penalty=1.0, pad_token_id=tok.eos_token_id,
            )
        gen = out[0][inputs.input_ids.shape[1]:]
        return tok.decode(gen, skip_special_tokens=True).strip()


def diff_str(a, b):
    diff = list(ndiff(a, b))
    diff_segs: list[str] = []
    i = 0
    while i < len(diff):
        d = diff[i]
        if not d.startswith(' '):
            j = i + 1
            last = d
            diff_str = last
            while j < len(diff) and not diff[j].startswith(' '):
                if diff[j][0] == last[0]:
                    diff_str += diff[j][-1]
                else:
                    last = diff[j]
                    diff_str += last
                j += 1
            pre = [d[-1] for d in diff[max(0, i - 5): i] if d.startswith(' ')]
            post = [d[-1]
                    for d in diff[j: min(len(diff) - 1, j + 5)] if d.startswith(' ')]
            diff_segs.append(''.join(pre + [f"({diff_str})"] + post))
            i = j
        i += 1
    return diff_segs

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def build_gbnf(sentence: str, tags: list[str] | None = None, word_break=" "):
    """
    Build a GBNF grammar that forces output to be a segmentation of EXACTLY the
    input characters, in order. The model may only choose (a) where word
    boundaries fall and (b) a POS tag per word (when `tags` is given). It cannot
    drop, add, or substitute any character -- so every 'char mismatch' failure
    (dropped punctuation, swapped rare glyphs, hallucinated text) becomes
    impossible at the decode step. Guarantees fidelity, NOT segmentation
    correctness. Pass tags=None for segmentation-only (no POS) output.
    """
    def esc(s):
        return s.replace("\\", "\\\\").replace('"', '\\"')

    seq = []
    chars = list(sentence)
    for i, ch in enumerate(chars):
        seq.append(f'"{esc(ch)}"')
        if i < len(chars) - 1:
            seq.append("gap")
    if tags:
        # closing tag of the final word
        seq.append("tag")
        return "\n".join([
            "root ::= " + " ".join(seq),
            # break: /TAG + space, or continue
            'gap ::= (tag " ") | ""',
            "tag ::= " + " | ".join(f'"/{t}"' for t in tags),
        ])
    return "\n".join([
        "root ::= " + " ".join(seq),
        f'gap ::= "{word_break}" | ""',
    ])


def load_model(*, model_id: str, device: str, arch: ARCH, api_base: str | None = None):
    """Lazy import so the parsing/QC logic can be used without torch installed."""
    if arch == "api":
        if not api_base:
            raise ValueError(
                "must provide api_base if using the api architecture")
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


@dataclass
class Tokenized:
    id: int
    chapter: str
    passage: str
    tokens: list[Word]

    def serialize(self) -> str:
        serialized = asdict(self)
        serialized['tokens'] = [w.serialize() for w in self.tokens if w.word]
        return json.dumps(serialized, ensure_ascii=False)


@dataclass
class TokenizedError(Tokenized):
    raw_output: str
    reason: str
    diffs: list[str]

    @classmethod
    def from_tokenized(cls, t: Tokenized, raw_output: str, reason: str, diffs: list[str]):
        return cls(t.id, t.chapter, t.passage, t.tokens, raw_output, reason, diffs)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Xunzi seg+POS for classical Chinese")
    ap.add_argument("process", choices=("seg", "segpos"),
                    help="whether to perform only word segmentation, "
                    "or word segmentation and part-of-speech analysis")
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
    ap.add_argument("--output", required=True,
                    help="output JSONL of tagged units")
    ap.add_argument("--errors", default=None,
                    help="JSONL for QC failures (default: <output>.errors.jsonl)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"],
                    help="'auto' uses GPU/MPS if available; 'cpu' forces CPU")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--grammar", action="store_true",
                    help="(api + llama.cpp only) constrain decoding to the exact input "
                         "characters via a per-sentence GBNF grammar; prevents char-mismatch errors")
    ap.add_argument("--limit", type=int, default=0,
                    help="tag only first N sentences (0 = all; use for a dry run)")
    args = ap.parse_args()

    errors_path = Path(args.errors) if args.errors else Path(
        str(args.output) + ".errors.jsonl")

    global API_MODEL, USE_GRAMMAR
    API_MODEL = args.api_model
    USE_GRAMMAR = args.grammar

    segpos: SegPos
    if args.process == "seg":
        segpos = SegPos(
            "seg",
            arch=assert_arch(args.arch),
            system_prompt=segmentation.seg.SYSTEM_PROMPT,
            fewshot=segmentation.seg.FEWSHOT,
            core_terms=[t.hanzi for t in TERMS]
        )
    elif args.process == "segpos":
        segpos = SegPos(
            "segpos",
            arch=assert_arch(args.arch),
            system_prompt=segmentation.segpos.SYSTEM_PROMPT,
            fewshot=segmentation.segpos.FEWSHOT,
            core_terms=[t.hanzi for t in TERMS]
        )
    else:
        raise ValueError(f"Bad process value: {args.process!r}")

    tok, model = load_model(
        model_id=args.model, device=args.device, arch=segpos.arch, api_base=args.api_base)

    mengzi = fetch_mengzi_full()
    lines: list[tuple[str, Chapter]] = []
    for chapter in mengzi.chapters:
        for line in chapter.text.split("\n"):
            lines.append((line, chapter))
    if args.limit:
        lines = lines[: args.limit]

    print(
        f"[info] {len(lines)}  line(s) to tag", file=sys.stderr)

    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(list(lines)), total=len(lines))
    except ImportError:
        iterator = enumerate(lines)

    n_ok = n_err = 0
    with open(args.output, "w", encoding="utf-8") as fout, \
            open(errors_path, "w", encoding="utf-8") as ferr:
        for i, (line, chapter) in iterator:
            raw = segpos.tag_sentence(tok=tok, model=model, sentence=line)
            tokens = segpos.parse(raw)
            ok, reason = segpos.qc_check(line, tokens)
            if not ok:
                # handle a common case where xunzi drops a trailing quote
                if tokens and reason == f"char mismatch: got {line[:-1]!r} vs {line!r}":
                    if line[-1:] == "」":
                        missing = Word("」")
                        if segpos.process == "segpos":
                            missing = Word("」", "w")
                        tokens.append(missing)
                        ok = True
                    elif line[-1:] == "』":
                        missing = Word("』")
                        if segpos.process == "segpos":
                            missing = Word("』", "w")
                        tokens.append(missing)
                        ok = True
                else:  # one stricter retry
                    raw = segpos.tag_sentence(
                        tok=tok, model=model, sentence=line, strict=True)
                    tokens = segpos.parse(raw)
                    ok, reason = segpos.qc_check(line, tokens)
            assert tokens is not None
            if ok:
                record = Tokenized(
                    id=i, chapter=chapter.title, passage=line, tokens=tokens)
                fout.write(record.serialize() + "\n")
                n_ok += 1
            else:
                record = TokenizedError(
                    id=i, chapter=chapter.title, passage=line, tokens=tokens,
                    raw_output=raw, reason=reason, diffs=diff_str(
                        ''.join([t.word for t in tokens]), line)
                )
                ferr.write(record.serialize() + "\n")
                n_err += 1

    print(
        f"[done] ok={n_ok}  needs_review={n_err}  -> {args.output}", file=sys.stderr)
    if n_err:
        print(f"[note] review failures in {errors_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
