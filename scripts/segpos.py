"""Generate a diagnostic segmentation + POS artifact for the Mengzi.

Reuses the same SuPar-Kanbun (spaCy) pipeline that feeds the live co-occurrence
build (nlp/chinese.py), tagging with Universal Dependencies UPOS. The output is
written for human inspection only — nothing in the site build reads it back.

    python segpos.py
"""
import json
from pathlib import Path

from mengzi import fetch_mengzi_full
from nlp.chinese import tag_segment, strip_punct

OUTPUT = Path(__file__).resolve().parent.parent / \
    "segpos" / "mengzi.segpos.jsonl"


def main() -> None:
    text = fetch_mengzi_full().text
    doc = tag_segment(strip_punct(text))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUTPUT.open("w", encoding="utf-8") as fout:
        for i, sentence in enumerate(doc.sents):
            tokens = [
                {
                    "word": t.text,
                    "upos": t.pos_,   # UD UPOS
                    "tag": t.tag_,    # SuPar-Kanbun fine-grained tag
                    "lemma": t.lemma_,
                    "gloss": t.norm_,  # English gloss
                }
                for t in sentence
                if t.text.strip()
            ]
            rec = {"id": i, "sentence": sentence.text, "tokens": tokens}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} sentences to {OUTPUT}")


if __name__ == "__main__":
    main()
