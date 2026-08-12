# Open decisions — Chinese retokenization

Things I could not settle without you, from the overnight session of
**2026-08-11 → 12** on `feat/chinese-retokenization`. Newest section last; each
item says what I did in the meantime so nothing is blocked on you.

Everything below is reversible. Nothing here has been merged to `dev`.

---

## 1. Re-run the segmentation with EvaHan few-shot examples?

**Built but unmeasured.** `segmentation/evahan.py` selects few-shot examples from
the EvaHan 2022 gold files, and `tools/eval_segmentation.py` scores word-span
P/R/F1 against held-out gold (shots from testa 左傳, evaluation on testb).

The case for it: EvaHan segments **exactly** the words the UD relations miss —
諸侯 163, 天下 109, 大夫 68, 寡人 62, 君子 27, 天子 25, 聖人 10 — so it demonstrates
the behaviour we want rather than merely tagging better.

The case against: the tags are never used downstream (the merge takes POS from
the treebank), so the only benefit is indirect, via better boundaries.

**What I did:** ran the overnight segmentation on the existing hand-written
few-shot, per your instruction to leave it. The EvaHan path is ready but unused.

**Decision needed:** worth ~10 min to score `evahan` vs `xunzi` vs zero-shot? If
EvaHan wins by a wide margin, a re-run costs ~40 min and every downstream number
shifts.

```
scripts/.venv/bin/python -m tools.eval_segmentation --n 120 --fewshot evahan
scripts/.venv/bin/python -m tools.eval_segmentation --n 120 --fewshot xunzi
```

---

## 2. `min_freq` needs recalibrating — your call where to cut

Merging moves the frequency distribution the embedding vocabulary is cut from,
so `MENGZI_PIPELINE.min_freq=5` is no longer tuned to anything. Measured with UD
relations only: vocab 602 → 606; 天 292 → 117, 下 227 → 47, 父 82 → 26, 母 47 → 10;
15 characters drop below the floor because they only ever occurred inside a
compound (倉, 廩, 姓, 禽, 畎 …). The lexicon moves it further.

**What I did:** left `min_freq=5` untouched — it is a tuning parameter and
`config.py` is yours.

---

## 3. Populate (or retire) `scripts/data/merge_overrides.json`

The override `merge` list existed to hand-add the words UD labels `nmod` —
諸侯, 天子, 大夫, 聖人. **The lexicon now supplies those**, so that list may no
longer be needed at all. `never_merge` is still the escape hatch for anything the
merge report shows as wrong.

**What I did:** shipped it with both lists empty.

**Decision needed:** review the merged-word list end to end — it is the
philological check, and nobody else can do it:

```
scripts/.venv/bin/python -m tools.merge_report --min-count 2
```

---

## 4. The stashed `never_merge` entry — reapply or drop?

`stash@{0}` holds your `"never_merge": ["者許行", "者夷之"]`. Those two come from a
genuine treebank error (`flat` attaching a name to the nominalizer 者 where
`appos` was meant — 4 such edges in the whole corpus).

**They are already inert**: both contain `PROPN` constituents, so `merged_pos`
labels them `PROPN` and `content_pos={"NOUN","VERB","ADJ"}` excludes them from
every output. The override only changes whether the merged token exists at all.

**Decision needed:** reapply for tidiness, or drop it as redundant.

---

## 5. Restrict the name rule to persons only?

`merged_pos` currently gives **any** group containing a `PROPN` the POS `PROPN`.
That catches place and state names as well as people — 齊國, 魯國, 岐山, 梁山, 幽州.
All seven are ≤2 occurrences, below both frequency floors, so it makes no
practical difference today.

To restrict it to persons, additionally require the root's XPOS to start with
`n,名詞,人` (王 公 伯 徒 = `人,役割`; 人 夫 = `人,人`; 子 弟 = `人,関係`), as against
`主体,集団` (國), `固定物,地形` (山), `制度,場` (州). One line, documented in
`merged_pos`.

**What I did:** kept the general rule, since it matches how the English side
excludes proper nouns wholesale.

---

## 6. Housekeeping

- `merge-report.txt` is untracked at the repo root (your run). Keep, gitignore,
  or delete?
- `segpos/conllu/` is untracked. It is generated output like `segpos/chapters/`,
  which _is_ committed — so it probably should be too, but it is ~1 MB of JSONL
  and I did not want to commit generated data without asking.
