"""Score a few-shot prompt's segmentation against EvaHan gold.

Motivation: the few-shot examples pin the segmentation convention, so changing
them changes the boundaries we consume — but there was no way to tell whether a
change helped. This measures it. Hold out one EvaHan gold file, run its lines
through the prompt, and score word-boundary P/R/F1 against gold.

**Sample shots from one file and evaluate on the other** — ``--shots`` defaults to
testa (左傳) and ``--gold`` to testb, so there is no leakage.

Evaluation is deliberately run on the *unpunctuated* view of the gold text, which
is the condition the Mengzi run actually operates in (the Kyoto CoNLL-U carries no
punctuation). Punctuation is a strong boundary cue, so scoring the punctuated text
would flatter every prompt equally and tell us nothing about the real task.

Metric: a predicted word counts as correct when its character span matches a gold
span exactly, which is the standard EvaHan segmentation criterion. The official
``eval_EvaHan_2022_FINAL.py`` in the ACDS repo cross-checks this, but wants its
inputs laid out as files in one directory; this computes it directly.

Run (needs the model served, e.g. `npm run xunzi:api`):
  scripts/.venv/bin/python -m tools.eval_segmentation --n 120
  scripts/.venv/bin/python -m tools.eval_segmentation --n 120 --fewshot xunzi
"""

from __future__ import annotations

import argparse
import sys

from segmentation import evahan, seg, segpos


def spans(words: list[str]) -> set[tuple[int, int, str]]:
    """Character spans of a word sequence: ``(start, end, surface)``."""
    out: set[tuple[int, int, str]] = set()
    i = 0
    for w in words:
        out.add((i, i + len(w), w))
        i += len(w)
    return out


def prf(gold: list[list[str]], pred: list[list[str]]) -> tuple[float, float, float]:
    """Micro-averaged precision / recall / F1 over exact word spans."""
    tp = n_gold = n_pred = 0
    for g, p in zip(gold, pred):
        gs, ps = spans(g), spans(p)
        tp += len(gs & ps)
        n_gold += len(gs)
        n_pred += len(ps)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return precision, recall, f1


def build_fewshot(kind: str, shots_path, n_shots: int) -> list[tuple[str, str]]:
    if kind == "none":
        return []
    if kind == "xunzi":
        return seg.unpunctuated_fewshot(seg.FEWSHOT)
    return evahan.select(shots_path, n=n_shots, tagset=segpos.VALID_TAGS,
                         unpunctuated=True, with_pos=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", default=str(evahan.TESTB),
                   help="gold file to score against (default: testb).")
    p.add_argument("--shots", default=str(evahan.TESTA),
                   help="gold file to draw EvaHan few-shot examples from "
                        "(default: testa). Must differ from --gold.")
    p.add_argument("--fewshot", default="evahan",
                   choices=["evahan", "xunzi", "none"],
                   help="evahan = selected gold lines; xunzi = the bundled "
                        "XunziALLM sample shots; none = zero-shot baseline.")
    p.add_argument("--n-shots", type=int, default=12, dest="n_shots")
    p.add_argument("--n", type=int, default=120,
                   help="gold lines to evaluate.")
    p.add_argument("--min-chars", type=int, default=18, dest="min_chars")
    p.add_argument("--max-chars", type=int, default=120, dest="max_chars")
    p.add_argument("--api-base", default="http://127.0.0.1:8080/v1")
    p.add_argument("--api-model", default="xunzi")
    p.add_argument("--no-grammar", action="store_true",
                   help="disable the GBNF constraint (then a prediction can "
                        "fail to round-trip and is scored as empty).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.gold == args.shots and args.fewshot == "evahan":
        sys.exit("--gold and --shots must differ, or the shots leak into the "
                 "evaluation set.")

    fewshot = build_fewshot(args.fewshot, args.shots, args.n_shots)
    gold_shots = [s.strip_punctuation() for s in evahan.load(args.gold)]
    gold_shots = [s for s in gold_shots
                  if args.min_chars <= len(s.text) <= args.max_chars][:args.n]
    print(f"prompt   : {args.fewshot} ({len(fewshot)} shots, "
          f"{sum(len(a) + len(b) for a, b in fewshot)} chars)", file=sys.stderr)
    print(f"eval set : {len(gold_shots)} lines from {args.gold}",
          file=sys.stderr)

    from cli import segment as seg_cli
    seg_cli.API_MODEL = args.api_model
    seg_cli.USE_GRAMMAR = not args.no_grammar
    runner = seg_cli.SegPos("seg", arch="api", fewshot=fewshot,
                            system_prompt=seg.SYSTEM_PROMPT)
    _, client = seg_cli.load_model(model_id="", device="cpu", arch="api",
                                   api_base=args.api_base)

    try:
        from tqdm import tqdm
        iterator = tqdm(gold_shots)
    except ImportError:
        iterator = gold_shots

    gold: list[list[str]] = []
    pred: list[list[str]] = []
    n_bad = 0
    for s in iterator:
        raw = runner.tag_sentence(tok=None, model=client, sentence=s.text)
        tokens = runner.parse(raw)
        words = [t.word for t in tokens] if tokens else []
        if "".join(words) != s.text:      # unusable; scored as a total miss
            n_bad += 1
            words = []
        gold.append(s.words)
        pred.append(words)

    p, r, f1 = prf(gold, pred)
    print(f"\n=== segmentation, exact word spans ({args.fewshot}) ===")
    print(f"  precision {p:.4f}   recall {r:.4f}   F1 {f1:.4f}")
    print(f"  gold words {sum(len(g) for g in gold)}, "
          f"predicted {sum(len(x) for x in pred)}, "
          f"lines failing round-trip {n_bad}")


if __name__ == "__main__":
    main()
