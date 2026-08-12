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

From the relations alone: **243 word types, 889 merged tokens** — 天下 ×173,
父母 ×37, 文王 ×35, 萬章 ×23, 百姓 ×19, 伊尹 ×19, 周公 ×18, 公孫丑 ×17, 禽獸 ×15.
With the segmenter lexicon (below) and the lexical-word guard, the total is
**730 types / 1907 tokens**.

### A merged name is a name

Inheriting the head's POS is right for ordinary compounds — 天下 is the `NOUN` 下
was — but wrong for a name. In a Chinese name+title compound the **modifier**
carries the referential identity and the head is an ordinary noun:

```
文/PROPN[NameType=Prs] --compound--> 王/NOUN   ("King Wen")
周/PROPN[NameType=Nat] --compound--> 公/NOUN   ("Duke of Zhou")
```

Head-inheritance would make those `NOUN`. Whenever `content_pos` is set to
`{"NOUN","VERB","ADJ"}` — as it is on the English side, precisely to keep proper
nouns out of the vocabulary — merging would then quietly re-admit 文王 ×35,
周公 ×18, 宣王 ×14, 武王 ×10, 繆公 ×9, 惠王 ×7 … , 29 name compounds clearing
`min_freq=5`, through a door the filter thought it had shut. Labelling them
`PROPN` keeps that decision in one place; whether the label excludes anything is
`content_pos`'s business (currently `None` for the Mengzi, so it excludes nothing).

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

### A nominalized predicate is a noun

Same problem, different construction. 者 nominalizes what precedes it — 賢者 "the
worthy one", 死者 "the dead" — but the treebank tags 者 `PART` and makes it the
group's head, so the merge inherits `PART` and the lexical-word guard throws it
away.

The fix is not "者 makes a noun", which would be wrong. **者's own dependency label
says which construction it is**, and the parse already records it:

```
賢者 = 賢/VERB/amod  + 者/PART/nsubj    "the worthy one"    nominal    -> NOUN
死者 = 死/VERB/amod  + 者/PART/obj      "the dead"          nominal    -> NOUN
儒者 = 儒/NOUN/nmod  + 者/PART/nmod     "a Confucian"       nominal    -> NOUN
昔者 = 昔/NOUN/nmod  + 者/PART/advmod   "in former times"   adverbial  -> refused
古者 = 古/NOUN/nmod  + 者/PART/advmod   "in ancient times"  adverbial  -> refused
```

`NOMINALIZERS` × `NOMINAL_DEPS` in `merged_pos` implements exactly that: +10 types
/ +26 tokens (賢者 8, 王者 5, 長者 5, 壯者 2, 死者, 老者, 儒者, 顯者, 使者, 生者), while
昔者 ×9, 或者 and 古者 stay refused as the adverbials they are.

**The general technique** — when a merged group inherits a function-word POS,
check the root's _slot_ rather than its tag — has one more plausible application,
not implemented: `NUM`-headed groups. 萬鍾 sits in `nsubj` and 百里 in `obl`/`root`,
which would make them nouns, whereas 萬乘, 千里, 五霸, 什一 sit in `nummod`, an
adnominal slot. Whether a measure phrase like 百里 "a hundred _li_" should be a
word at all is a judgement call rather than a parsing question, so it is left
alone. 15 tokens.

## What we do not merge, and why

**`conj` and `nmod` are excluded.** This is the load-bearing decision. 仁義 is
義 --`conj`--> 仁; merging it would eat occurrences of two target terms. `nmod`
is worse — it is the relation on genuine modifiers, so adding it over-merges
broadly.

The cost is that the treebank labels several real bisyllabic words `nmod` and the
relations therefore miss them: **諸侯, 天子, 大夫, 聖人, 寡人, 庶人, 小人**. Those are
supplied by the segmenter lexicon instead — see "The second boundary source".

Widening the relation set to include `nmod` was measured and rejected. It takes
the corpus from 243 to 594 types (889 → 1529 tokens) and fails three ways: it
turns phrases into nodes (方百 ×7, from 方百里 "a hundred _li_ square"; 民父母 ×5,
from 為民父母; 人者 ×5), it **extends words that were already right** (父母 →
民父母 ×5 and 父母國 ×2; 天下 → 天下諸侯 ×2; 百姓 → 百姓者), and it swallows six more
target occurrences. `nmod` is a phrase-level relation — the treebank uses the same
label for 諸侯 and for 民父母 — so it cannot answer the question we would be asking
it.

### Guards

Applied to every candidate group, whatever source proposed it. Counts are for
the full run (relations + lexicon):

| Guard                           | Rejects                                     | Why                                                                                                                                                        |
| ------------------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Contiguous run only             | 8 — 欣然, 子男, 民夫婦                      | The relation reaches across an intervening token; merging would reorder the text                                                                           |
| Single sentence                 | 1                                           | Structural safety; a merge must not cross a sentence boundary                                                                                              |
| Begins/ends on a token boundary | 5                                           | Lexicon only: the segmenter may split a token the treebank keeps whole (孟子 → 孟 + 子). Skipped, never forced                                             |
| All constituents are stopwords  | 161 — 可以 ×77, 以為 ×12, 而已 ×12, 得而 ×7 | Each character is already dropped downstream, so _not_ merging reproduces current behaviour. Merging would invent a 可以 content node out of two stopwords |
| Contains a target hanzi         | 23 — 禮貌, 仁政, 仁人, 智慧, 禮義           | 仁 義 禮 智 信 keep every occurrence, so counts stay comparable across runs. Logged in the report, never silently swallowed                                |
| **Result is a lexical word**    | 157 — 然後, 足以, 昔者, 之心, 萬乘          | The group's inherited POS must be `NOUN`/`PROPN`/`VERB`/`ADJ`. See below                                                                                   |
| `never_merge` override          | curated                                     | See below                                                                                                                                                  |

The target guard is what makes the segmenter safe to consult: its own convention
merges 仁政, 仁人 and 智慧, and all 23 such groups are refused, so the shipped
occurrence counts are byte-identical to the unmerged baseline.

### Why a merge must produce a lexical word

`MergeConfig.mergeable_pos` refuses any group whose inherited POS is a function
word — `SCONJ` (之心, 之人), `PART` (昔者, 賢者), `AUX` (足以, 敢以), `ADV` (然後,
沛然), `NUM` (萬乘, 萬鍾). 157 tokens across 52 types.

This is not a filter on what gets analysed; it is a statement about **word
formation**, and it must stay independent of `Pipeline.content_pos` (a tuning knob
that may be `None`). The reason is that merging is destructive in both directions:
a merged token _replaces_ its constituents, so forming 之心 buries 心 inside a
token that is not a word and that nothing downstream wants either — 心 loses the
occurrence and nothing gains it. Refusing the merge lets 之 be dropped as a
stopword and 心 stand as itself, which is what 惻隱之心 needs if the 仁–心 relation
is to survive.

Measured, the rule restores 足 +35, 後 +21, 昔 +11, 心 +7, 賢 +7, 長 +6, 人 +6,
萬 +6, 惡 +4. `PROPN` stays mergeable on purpose: a name **is** a lexical word, and
whether names are analysed is `content_pos`'s business, not this rule's.

Totals with the rule: **730 types / 1907 tokens**.

## Curated overrides

`scripts/data/merge_overrides.json` (optional — a missing file is not an error):

- **`merge`** force-merges a run of whole tokens spelling the word, anywhere it
  appears inside one sentence. Because listing a word is an explicit judgement, a
  forced group bypasses the stopword and target guards (it still must be
  contiguous and within one sentence). This was the intended home for
  諸侯 / 天子 / 大夫 / 聖人 before the lexicon existed; **the lexicon now supplies
  those**, so the list may not be needed at all.
- **`never_merge`** drops a word the other sources do produce. Checked first, so
  listing a word in both means it is not merged — an explicit refusal beats an
  explicit request.

Both match either the joined surface forms or the joined lemmas, so a word can be
listed with whichever glyphs are to hand (荅 or 答).

Populate it from `scripts/.venv/bin/python -m tools.merge_report`, which lists
every merged type with its inherited POS, every guard rejection by reason, and
the characters losing the most occurrences to a merge.

## Consequences to watch

**`Pipeline.min_freq` shifts under you.** The embedding vocabulary at
`min_freq=5` goes 602 → 606 with the relations alone, and **602 → 571** with the
lexicon as well. 天 drops 292 → 117, 下 227 → 47, 父 82 → 26, 母 47 → 10, while
天下 (173) and 父母 (37) appear as words; characters that only ever occurred
inside a compound (倉, 廩, 姓, 禽, 畎 …) fall below the floor entirely. The
distribution the vocabulary is cut from is not the one `min_freq=5` was tuned
against — re-check it after any change to the merge set. (Target counts are
unchanged by construction.)

**`cooccurrence_min_freq` deserves the same look.** Per-chapter networks thin
slightly, because a chapter's characters are now spread across fewer, longer
types: of 81 network files, 74 keep their node count, 6 lose nodes (義\_3A 16 → 6
is the largest) and none becomes empty. The `智`/`信` chapters that log "no
co-occurrence" were already empty before merging.

**Merged spans embed as spans.** GujiRoBERTa is character-tokenized, so a
two-character word is two subwords pooled by `subword_pooling="mean"` (already
the default). Character offsets are untouched by retokenization — `doc.text` and
every `token.idx` survive it — so the segment math in `embeddings.occurrences`
needs no change.

**The front end already handles this.** `src/lib/build/cedict.ts` looks
multi-character headwords up in CC-CEDICT and falls back to per-character pinyin
for words it lacks (許子, 瞽瞍).

## The second boundary source: a segmenter lexicon

The UD relations cannot reach the bisyllabic words the treebank labels `nmod`.
Those come from a **segmentation of the treebank's own text** — XunziALLM via
`cli.segment seg --source conllu`, written to `segpos/conllu/mengzi.seg.jsonl`
and merged by `corpus.recombine.lexicon_groups`.

### Why there is no alignment step

An earlier prototype aligned the pre-existing `segpos/chapters/*.jsonl` (produced
from the **ctext** edition) against the treebank with `difflib`. It worked — the
editions are 99.4% character-identical — but reconciling two editions inside the
pipeline is not something you should have to defend.

Segmenting the treebank's own text removes the problem rather than solving it.
`corpus.conllu.iter_units` packs treebank sentences into units under
`--max-chars` that never cross a `# newpar`, and each unit records `doc_id`,
`sent_ids`, `token_start` and `n_tokens`. Because a unit names the exact token
range it covers, mapping the segmentation back is a **direct index map**. The two
sources describe the same string; nothing is aligned, matched or guessed.

A segmenter word is used only when it begins and ends on treebank token
boundaries. The segmenter is free to split a token the treebank keeps whole
(孟子 → 孟 + 子); such a word is skipped, never forced. Over the whole corpus that
cost 5 words.

### The run

442 units (mean 80 characters), **442 succeeded, 0 errors, 0 round-trip
failures**, covering all 34,289 tokens. Decoding is constrained by a per-unit
GBNF grammar that permits only the input characters, so character mismatch — the
dominant failure mode of the earlier ctext runs — is structurally impossible.

Two prompt details matter for reproducing it. The treebank text is
**unpunctuated**, so the few-shot examples must be too (`FEWSHOT_UNPUNCTUATED`,
`seg.unpunctuated_fewshot`); demonstrating a `w` token teaches output the input
cannot support. And paragraph-sized units are the right granularity — per-sentence
would be far worse, since treebank sentences have a median of 5 characters and
44% are ≤ 4.

### What each source contributes

| source         | tokens | types | examples                                |
| -------------- | -----: | ----: | --------------------------------------- |
| lexicon only   |   1154 |   523 | 諸侯 天子 大夫 聖人 寡人 小人 大人 庶人 |
| both agree     |    732 |   206 | 天下 文王 父母 萬章 百姓 伊尹 公孫丑    |
| relations only |    178 |    95 | 足以 牛羊 上下 管仲 鄉原 父兄 父子 左右 |

**777 word types / 2064 tokens** in total. The split is the argument for keeping
both: the segmenter finds 523 types the relations cannot, and the relations still
find 95 the segmenter does not. Neither subsumes the other.

Guards reject 161 all-stopword groups (可以 ×77, 以為 ×12, 而已 ×12 …), 23 groups
containing a target, 8 non-contiguous, 5 that split a treebank token, and 1
crossing a sentence. The target count rises from 1 to 23 with the lexicon active —
the segmenter's convention merges 仁政, 仁人, 智慧 — and every one is refused, which
is why the shipped occurrence counts are identical to the unmerged baseline
(仁 158, 義 108, 禮 68, 智 32, 信 30).

The lexicon is **optional**: generating it is a manual step, so
`Pipeline.merge_lexicon` defaults to `None` and a missing file is a no-op, never
an error. `segpos/chapters/*.jsonl` (the ctext-derived output) is left in place
and unused — its records carry no `token_start`, so `load_lexicon` skips them.

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
