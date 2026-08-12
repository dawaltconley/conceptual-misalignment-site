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
few-shot, per your instruction to leave it. Then, since measuring needs no input
from you (only acting on it does), I scored all three on 150 held-out testb lines
(5321 gold words), decoding grammar-constrained, 0 round-trip failures:

| few-shot        | precision | recall |     F1 |
| --------------- | --------: | -----: | -----: |
| **evahan** (12) |    0.9253 | 0.9434 | 0.9343 |
| xunzi (70)      |    0.9141 | 0.9404 | 0.9271 |
| none (0)        |    0.9256 | 0.8927 | 0.9088 |

**EvaHan wins by +0.7 F1 over the shots we actually used**, with a prompt 2.4×
smaller (12 shots / 1316 chars vs 70 / 3135). Zero-shot has the best precision but
much the worst recall — it under-segments (5132 predicted against 5321 gold),
which is exactly the failure the few-shot examples are there to fix.

Overall F1 is a misleading headline, though, because it is dominated by
single-character words every prompt gets right. **The merge only ever consumes
multi-character boundaries**, so I added that breakdown and re-ran:

| few-shot        | multi-char P | multi-char R | multi-char F1 |
| --------------- | -----------: | -----------: | ------------: |
| **evahan** (12) |       0.8643 |   **0.7829** |    **0.8216** |
| xunzi (70)      |       0.8724 |       0.7378 |        0.7995 |

On the metric that matters the gap is **+2.2 F1, three times the overall gap, and
it is entirely recall** (+4.5 points) at essentially tied precision. Of 797
multi-character gold words, EvaHan shots find ~36 more per 150 lines — a ~6%
relative gain in words found. Scaled to the Mengzi's 2847 multi-character word
tokens that is on the order of 170 more words, which is material for a source
whose entire purpose is catching what the relations miss.

For calibration: re-running `xunzi` gave 0.9279 against 0.9271 the first time, so
~0.001 is run-to-run noise. +0.022 is well clear of it.

> **My recommendation: re-run the segmentation with the EvaHan few-shot.** I did
> not do it, because you said to leave the overnight run alone — and it also
> invalidates the committed pipeline output, which would want re-verifying.

```
# ~30 min; then re-run the pipeline and re-check the merge report
scripts/.venv/bin/python -m cli.segment seg --source conllu --arch api \
  --api-base http://127.0.0.1:8080/v1 --grammar \
  --fewshot evahan --output ../segpos/conllu/mengzi.seg.jsonl
```

The `--fewshot` flag is wired and smoke-tested, so that command runs as written.
If you would rather I just do it end to end — re-segment, re-run the pipeline,
re-verify and re-commit — say so and it is one instruction.

The one caveat that survives: the evaluation is on **資治通鑑 narrative** (testb)
with shots from **左傳** (testa), while the Mengzi is philosophical dialogue. The
selector already scores candidates to prefer common-noun compounds and dialogue
particles over proper-noun-dense narrative, but the measured gain is still
out-of-domain evidence.

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

## 5b. `content_pos` is commented out in `MENGZI_PIPELINE` — deliberate?

Uncommitted in `config.py`: `# content_pos=frozenset({"NOUN", "VERB", "ADJ"})`,
so it is `None` and **no POS filtering happens on the Chinese side**. Flagging it
because it interacts with two things above:

- Proper nouns are no longer excluded, so 文王, 周公, 齊宣王 … now become nodes.
  The `merged_pos` rule (item 5) still labels them `PROPN`; nothing acts on it.
- It is why the "don't merge non-lexical groups" rule is written against its own
  `mergeable_pos` set rather than against `content_pos` — tying word formation to
  a filter that may be `None` would have made the rule vanish exactly when you
  turned the filter off.

No action taken; just make sure it is intended before the next run.

---

## 5c. Should measure phrases be words? (`NUM`-headed merges)

The 者 nominalizer rule generalises: when a merged group inherits a function-word
POS, its root's _dependency slot_ often says what it really is. The remaining
candidate is `NUM`-headed groups, 15 tokens:

- **nominal slots** — 萬鍾 (`nsubj`), 百里 (`obl`, `root`), 什一 (`root`). Applying
  the same rule would make these nouns and merge them.
- **adnominal slots** — 萬乘, 千里, 五霸, 什一 (`nummod`). Would stay refused.

I did not implement it, because unlike 者 this is not a parsing question. 百里 "a
hundred _li_" is a measure phrase; whether it should be a vocabulary node is a
philological call. 五霸 "the Five Hegemons" is arguably a proper noun and a
different case again.

Everything else refused as non-lexical looks correctly refused: `ADV` 66 (然後,
沛然, the whole X然 adverbial family), `AUX` 43 (足以, 敢以), `SCONJ` 11 (之心, 之人),
`PART` 14 (昔者 ×9 and friends), plus 由此 (`PRON`) and 而後 (`CCONJ`).

---

## 6. Housekeeping

- `merge-report.txt` is untracked at the repo root (your run). Keep, gitignore,
  or delete?
- `segpos/conllu/` is untracked. It is generated output like `segpos/chapters/`,
  which _is_ committed — so it probably should be too, but it is ~1 MB of JSONL
  and I did not want to commit generated data without asking.

---

## 7. Review the 330 words the segmenter added (the philological check)

The lexicon contributes **330 types that reach the vocabulary as `NOUN`**, 80 of
them with ≥3 occurrences. This is the part only you can sign off on. The list is
strong on inspection — 諸侯 58, 天子 35, 大夫 29, 聖人 25, 寡人 22, 小人 13, 大人 12,
庶人 11, 妻子 9, 中國 8, 丈夫 8, 良人 7, 先王 6, 匹夫 6, 有司 6, 人倫 6, 上士/中士/下士 6,
四海 5, 赤子 5, 天爵 5, 人爵 5, 社稷 5, 人心 4, 經界 4, 規矩 4, 條理 4, 宗廟 3, 孝子 3,
人性 3 — and it includes **杞柳 5** and **桮棬 5**, the two words you originally
asked about.

```
scripts/.venv/bin/python -m tools.merge_report --min-count 3 --pos NOUN
```

Specific things I would check:

- **地方 ×4** looks like an over-merge. Mengzi's phrase is 地方百里 "territory a
  hundred _li_ square", i.e. 地 + 方百里, not the modern compound 地方 "place".
  A `never_merge` candidate.
- Temporal/measure expressions are borderline rather than wrong: 今日 9, 前日 6,
  他日 4, 三年 5, 來年 3, 百世 3, 終身 3, 尺寸 3.
- 大王 ×3 — is this the title, or 太王 (King Tai)? The treebank normalises 大/太
  inconsistently (see the note's edition section).

Two clear segmenter errors are **already harmless**: 之心 ×7 and 之人 ×4 (merging
the genitive 之) come out `SCONJ`, and 昔者 ×11 / 賢者 ×7 / 長者 ×6 come out `PART`,
so `content_pos` discards all of them. POS inheritance is doing real work here.

---

## 8. `cooccurrence_min_freq` may want lowering too

Same root cause as `min_freq` (item 2), separate knob. Per-chapter networks thin
because a chapter's characters now spread across fewer, longer types. Of 81
network files: **74 unchanged, 6 with fewer nodes, 0 empty, 0 with more**. The
largest drop is 義\_3A 16 → 6. The `智`/`信` chapters that log "no co-occurrence"
were already empty before merging, so that is not a regression.

`cooccurrence_min_freq=3` is already low; going to 2 would partly compensate but
also admits more noise. Your call.

---

## 9. Pipeline output has been regenerated

`scripts/.venv/bin/python -m main --corpus mengzi` ran clean in 10.2s and its
output is committed separately so it is easy to revert. 67 files under
`public/ctext/`, plus `public/embeddings/mengzi.json` and `src/data/terms.json`.

Verified: target occurrence counts in `terms.json` are unchanged (仁 158, 義 108,
禮 68, 智 32, 信 30); `npm run build` succeeds; 41 distinct multi-character node
ids now appear across the shipped networks. The embedding vocabulary is 571 nodes
(was 602 unmerged), 2860 edges, 15 communities.

**Not re-run: the SEP corpus.** Nothing in this work touches it, and
`build_master` composed `terms.json` from the existing manifest.
