# Multi-character tokenization for the Mengzi

How the Chinese corpus gets words instead of characters, what is deliberately
_not_ merged, and where the remaining judgement calls live.

_Written 2026-08-11._

## The problem

The Kyoto UD treebank (`scripts/data/mengzi.conllu`) annotates classical Chinese
**one character per token**. Taken at face value that makes the corpus vocabulary
characters rather than words: 天下 counts as 天 + 下, 父母 as 父 + 母, 文王 as 文 + 王.
Two things go wrong downstream:

- **Frequency inflation.** 天 accumulates 173 occurrences it only has as part of
  天下 — generic glyphs float to the top of the co-occurrence networks and
  dominate the embedding vocabulary's frequency gradient.
- **Split occurrences.** One word's contexts are divided across two nodes, so
  neither node's vector is a good estimate of anything.

## What we merge

The treebank already records which adjacent tokens form a single word: the UD
**word-formation relations**. `Pipeline.merge_deps` on `MENGZI_PIPELINE` is set to

```python
frozenset({"compound", "flat", "fixed"})
```

Tokens connected — in either direction — by one of these are unioned and
recombined into one token by `corpus.recombine.merge_doc`, applied inside the
CoNLL-U loader so nothing downstream has to know the corpus was ever characters.
The merged token's **lemma is the joined lemmas** (node ids key on the lemma) and
its **text is the joined forms** (the display glyph, which
`cooccurrence.pmi_spacy.attach_forms` already surfaces). It inherits tag, morph
and dep from the group's syntactic root, and its POS from the root **except for
names** — see below.

Result over the whole Mengzi: **243 word types, 889 merged tokens** — 天下 ×173,
父母 ×37, 文王 ×35, 萬章 ×23, 百姓 ×19, 伊尹 ×19, 周公 ×18, 公孫丑 ×17, 禽獸 ×15.

### A merged name is a name

Inheriting the head's POS is right for ordinary compounds — 天下 is the `NOUN` 下
was — but wrong for a name. In a Chinese name+title compound the **modifier**
carries the referential identity and the head is an ordinary noun:

```
文/PROPN[NameType=Prs] --compound--> 王/NOUN   ("King Wen")
周/PROPN[NameType=Nat] --compound--> 公/NOUN   ("Duke of Zhou")
```

Head-inheritance would make those `NOUN`, and since `content_pos` is
`{"NOUN","VERB","ADJ"}` precisely to keep proper nouns out of the vocabulary
(the English side excludes them the same way), merging would quietly re-admit
文王 ×35, 周公 ×18, 宣王 ×14, 武王 ×10, 繆公 ×9, 惠王 ×7 … — 29 name compounds
clearing `min_freq=5`.

So `corpus.recombine.merged_pos` gives a group containing a proper noun the POS
`PROPN`. **The treebank's gold annotation is the entire source — no NER pass is
needed.** 141 types / 386 tokens take PROPN this way, and every compound with no
`PROPN` constituent (天下, 父母, 禽獸, 土地, 百姓, 國家, 君臣, 倉廩 …) is untouched.

The rule catches place and state names too — 齊國, 魯國, 岐山, 梁山, 幽州 — because
their modifier is `PROPN` as well. That is consistent with how the English side
treats proper nouns, and academic: all seven are ≤2 occurrences, below both
frequency floors. To restrict the rule to **persons** specifically, additionally
require the root's XPOS to start with `n,名詞,人` — the treebank's person
categories (王 公 伯 徒 = `人,役割`; 人 夫 = `人,人`; 子 弟 = `人,関係`) as against
`主体,集団` (國), `固定物,地形` (山) and `制度,場` (州).

## What we do not merge, and why

**`conj` and `nmod` are excluded.** This is the load-bearing decision. 仁義 is
義 --`conj`--> 仁; merging it would eat occurrences of two target terms. `nmod`
is worse — it is the relation on genuine modifiers, so adding it over-merges
broadly.

The cost is that the treebank labels several real bisyllabic words `nmod` and we
therefore miss them: **諸侯, 天子, 大夫, 聖人, 寡人, 庶人, 小人**. This is a known
gap, not an oversight; see "Curated overrides" and "Deferred" below.

### Guards

Applied to every candidate group, with the corpus-wide counts they reject:

| Guard                          | Rejects                                           | Why                                                                                                                                                        |
| ------------------------------ | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Contiguous run only            | 29 — 欣然, 由然, 子男, 民夫婦                     | The relation reaches across an intervening token; merging would reorder the text                                                                           |
| Single sentence                | 0                                                 | Structural safety; a merge must not cross a sentence boundary                                                                                              |
| All constituents are stopwords | 96 — 可以 ×76, 得而 ×7, 有以 ×5, 無以 ×2, 能以 ×2 | Each character is already dropped downstream, so _not_ merging reproduces current behaviour. Merging would invent a 可以 content node out of two stopwords |
| Contains a target hanzi        | 1 — 禮義                                          | 仁 義 禮 智 信 keep every occurrence, so counts stay comparable across runs. Logged in the report, never silently swallowed                                |
| `never_merge` override         | curated                                           | See below                                                                                                                                                  |

Note what the guards do **not** have to handle. Merged tokens inherit their
root's POS, so 足以 comes out `AUX` and the 欣然 family `ADV`; the pipeline's
existing `content_pos={"NOUN","VERB","ADJ"}` discards those 72 of 889 with no
extra rule. The final split is 431 `NOUN`, 386 `PROPN` (names, per the rule
above), 40 `AUX`, 18 `VERB`, 14 `ADV` — so only the 431 `NOUN` merges reach the
vocabulary.

## Curated overrides

`scripts/data/merge_overrides.json` (optional — a missing file is not an error):

- **`merge`** force-merges a run of whole tokens spelling the word, anywhere it
  appears inside one sentence. This is where 諸侯 / 天子 / 大夫 / 聖人 belong: a short
  list you can defend line by line, rather than a relation set loosened until it
  happens to catch them. Because listing a word is an explicit judgement, a forced
  group bypasses the stopword and target guards (it still must be contiguous and
  within one sentence).
- **`never_merge`** drops a word the relations do produce.

Both match either the joined surface forms or the joined lemmas, so a word can be
listed with whichever glyphs are to hand (荅 or 答).

Populate it from `scripts/.venv/bin/python -m tools.merge_report`, which lists
every merged type with its inherited POS, every guard rejection by reason, and
the characters losing the most occurrences to a merge.

## Consequences to watch

**`Pipeline.min_freq` shifts under you.** Measured on the embedding vocabulary at
`min_freq=5`: 602 → 606 words. 天 drops 292 → 117, 下 227 → 47, 父 82 → 26, 母
47 → 10, while 天下 (173) and 父母 (37) appear as words. Fifteen
characters fall _below_ the floor because they only ever occurred inside a
compound (倉, 廩, 姓, 禽, 畎 …). The distribution the vocab is cut from is not the
one `min_freq=5` was tuned against — re-check it after any change to the merge
set. (Target counts are unchanged by construction: 仁 156, 義 108.)

**Merged spans embed as spans.** GujiRoBERTa is character-tokenized, so a
two-character word is two subwords pooled by `subword_pooling="mean"` (already
the default). Character offsets are untouched by retokenization — `doc.text` and
every `token.idx` survive it — so the segment math in `embeddings.occurrences`
needs no change.

**The front end already handles this.** `src/lib/build/cedict.ts` looks
multi-character headwords up in CC-CEDICT and falls back to per-character pinyin
for words it lacks (許子, 瞽瞍).

## Deferred: a second boundary source

The XunziALLM segmentation under `segpos/` catches exactly the words UD labels
`nmod` — an aligned prototype found ~1,300 merges the relations miss (諸侯 ×57,
天子 ×35, 大夫 ×29, 聖人 ×27, 寡人 ×23). It is **not** wired into the pipeline, and
nothing here depends on `segpos/` existing: that output is a manual step, and the
existing files were produced from the **ctext** edition rather than the treebank,
so using them means reconciling two editions.

Measured facts about that reconciliation, kept here so they need not be
re-derived:

- The editions are **99.4% character-identical** compared on the CoNLL-U `form`
  column. Diffing on `form` already absorbs the big normalizations (爲→為 512×,
  吿→告 41×, 敎→教 35×, 郷→鄉 27×) — they never appear as differences at all.
- Normalization is not uniformly modern-ward: lemma 答 → form 荅 (6×), lemma 間 →
  form 閒 (2×), lemma 疏 → form 䟽 (2×). A comparison should count a character as
  matching if **either** the form or the lemma matches ctext.
- The residual is genuine edition variance, and it is the part worth hand-review:
  **144 single-character substitutions** over 80 distinct pairs (間/閒 ×12, 荅/答 ×6,
  歟/與 ×6, 絜/潔 ×5, 脩/修 ×5), about **20 single-character insert/deletes**, a few
  doubled-graph variants (鶂鶂/鶃鶃, 昏昏/昬昬), one 小子/士, and **one real gap** — a
  34-character passage in 3A that ctext has and the treebank lacks
  (是亂天下也巨屨小屨同賈人豈為之哉…).
- Units do not correspond: 260 `# newpar` paragraphs in the treebank (mean 136
  chars, max 1313) against 690 ctext passages in `segpos/chapters/*.jsonl`.

The agreed path is to re-run the segmentation against the **treebank** text into a
fresh `segpos/` subdirectory, flag only the passages where the two source texts
genuinely differ, and hand-merge those — so the final boundary set is curated
rather than machine-aligned. Caveat for that run: the treebank text is
**unpunctuated**, while the prompt and few-shot examples in
`scripts/segmentation/segpos.py` assume punctuation down to the `w` tag.
Paragraph units are the right granularity; per-sentence would be worse (median 5
characters, 44% ≤ 4).

`corpus.recombine` is built for this: sources contribute `(i, j)` index pairs that
are unioned before the guards run, so a lexicon source joins in without touching
the merge logic. Its input file being absent must stay a no-op, never an error.

## References

- UD Classical Chinese (Kyoto) — relation inventory and the `compound`/`flat`/
  `fixed` definitions: <https://universaldependencies.org/treebanks/lzh_kyoto/>
- `notes/tagger-comparison.md` — why the gold treebank rather than a parser; both
  suparkanbun and CLTK label 天子/諸侯 `nmod` too, so swapping parsers would not
  close the gap above.
- `notes/embedding-communities-and-semantics.md` — what the embedding vocabulary
  is actually clustering, which the merge changes the frequency profile of.
- Prior art: `git show suparkanbun-cltk-comparison:scripts/nlp/recombine.py`
  (never merged) — the original grouping algorithm, written against the
  suparkanbun parse.
