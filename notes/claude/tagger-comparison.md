# Classical-Chinese tagger comparison — SuPar-Kanbun vs CLTK

Segmentation + POS-tagging on the *Mengzi*, comparing the two taggers wired into the
pipeline: **SuPar-Kanbun** (`roberta-classical-chinese-base-char`, spaCy) used by
`scripts/nlp/chinese.py` + `scripts/segpos.py`, and **CLTK** (Stanza `kyoto`, Literary
Chinese) used by `scripts/segpos_cltk.py`.

Interactive report: <https://claude.ai/code/artifact/571ca27b-a33c-4b6f-8f74-7bbd0a495ca8>

_Generated 2026-07-15._

## Gold-standard evaluation (the headline)

Scored against `UD_Classical_Chinese-Kyoto/mengzi.conllu` — 6568 sentences, raw tokens
(**no subword recombination**, since the gold keeps compounds like 惠王 split),
punctuation excluded, span-based P/R/F1 on UD UPOS. Both taggers aligned all 6568
sentences (0 skipped).

| Metric | **SuPar-Kanbun** | **CLTK (Stanza kyoto)** |
|---|---|---|
| Segmentation F1 | **99.69** | 99.12 |
| UPOS F1 | **94.43** | 91.87 |
| UPOS accuracy on correctly-segmented tokens | **94.73%** | 92.69% |

**SuPar-Kanbun leads on every metric** — segmentation is near-perfect for both; the real
gap (~2.5 UPOS F1) is in POS tagging.

### ⚠️ Caveat: reproduction, not accuracy
Both models were trained on UD Classical Chinese — CLTK's `kyoto` model on **this very
treebank** — so the Mengzi sentences are almost certainly in-training. These numbers
measure how faithfully each reproduces the Kyoto scheme, not accuracy on unseen text. The
head-to-head is fair, but absolute figures are optimistic. Notably, SuPar-Kanbun still
wins despite CLTK's home-field advantage. To get a truer estimate, re-run against the
Kyoto **dev/test** split instead of the full Mengzi.

### Where each one slips (UPOS recall by gold tag)
- **SuPar-Kanbun** — excellent except two concentrated spots: **ADV 76.4%** and
  **AUX 78.2%**, almost all from reading gold adverbs/auxiliaries as `VERB`
  (`ADV→VERB` 863×, `AUX→VERB` 151×). This is the classic Classical-Chinese
  stative-verb / adverb boundary — an annotation-convention disagreement more than an error.
- **CLTK** — softer and more broadly: **PROPN 83.3%** (proper names → `NOUN` 104×; its
  PROPN *segmentation* recall is also only 92.3% — mis-splits multi-char names), plus
  VERB 90.5%, ADP 90.5%, AUX 85.6%. Top confusions spread across open classes
  (`VERB→NOUN` 522×, `ADV→VERB` 386×, `NOUN→VERB` 331×).

**Takeaway:** SuPar-Kanbun trades one large systematic ADV/AUX→VERB slip for stronger
scores everywhere else; CLTK has no single big failure but trails across the board.

## Head-to-head (no gold needed)

Both run over the full Mengzi **with** the shared subword recombination
(`scripts/nlp/recombine.py`), aligned on CJK characters (punctuation excluded), 2890
sentences:

- **Segmentation agreement: 93.6%** (of word spans); 635 sentences differ on ≥1 boundary.
- **POS agreement: 90.7%** on identically-segmented tokens.
- Making CLTK recombine too *lowered* raw agreement (was 98.2% when CLTK stayed
  character-level) — because it exposes where the two **dependency parsers disagree about
  compound structure** rather than a merge/no-merge asymmetry.
- Systematic mutual POS splits mirror the gold findings: `VERB↔ADV/NOUN/AUX`, and
  convention differences `之` PRON↔SCONJ, `乎` PART↔ADP.

## Practical notes for this project
- **SuPar-Kanbun preserves source characters exactly** (char-exact segmentation); CLTK
  rewrites full-width punctuation to half-width — relevant since the site keys nodes and
  CC-CEDICT lookups on the surface string. (Moot for gold eval; gold has no punctuation.)
- SuPar-Kanbun is the tagger feeding the live co-occurrence build. CLTK is diagnostic
  only (`cltk` is not in `requirements.txt`; `pip install cltk` to run `segpos_cltk.py`).

## Reproduce
```bash
cd scripts
python segpos.py          # SuPar-Kanbun -> segpos/mengzi.segpos.jsonl
python segpos_cltk.py     # CLTK         -> segpos/mengzi.segpos.cltk.jsonl  (needs: pip install cltk)
```
Gold evaluation was run with an ad-hoc script (span-based P/R/F1 over `mengzi.conllu`,
raw tokens, punctuation dropped). Not yet committed — ask to add it as
`scripts/eval_gold.py` (parameterized by conllu path) if you want it reproducible.
