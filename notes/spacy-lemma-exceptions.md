# spaCy lemma errors in the SEP corpus, and the exception table

`en_core_web_sm` makes two systematic lemmatization errors on philosophical
English. Both were found while investigating scatter clutter (see
[[derivational-variant-merging]]) and both are fixed by
`scripts/lemmas/english.conf`, applied in `corpus.parse._apply_lemma_exceptions`.

Measured over the parsed SEP corpus (182 articles, `min_freq=20`, SEP content
POS `{NOUN, VERB, ADJ}`), not guessed.

## Error 1 — discipline names in `-ics` lose their final s

spaCy treats `-ics` as a plural marker, so a field name is demoted to a
singular that is usually a *different, real word*. The damage is that the field
noun then gets pooled together with an unrelated adjective:

| surface | spaCy lemma | occurrences of the lemma that were the `-ics` form |
|---|---|---|
| `ethics` | `ethic` | 813 of 848 (96%) |
| `semantics` | `semantic` | 344 of 918 (37%) |
| `aesthetics` | `aesthetic` | 206 of 536 (38%) |
| `politics` | `politic` | 190 of 196 (97%) |
| `mathematics` | `mathematic` | 164 of 164 (100%) |
| `metaphysics` | `metaphysic` | 101 of 106 (95%) |
| `economics` | `economic` | 79 of 463 (17%) |
| `metaethics` | `metaethic` | 40 of 42 (95%) |
| `physics` | `physic` | 33 of 33 (100%) |

**This cannot be automated.** There is no signal separating a pluralia-tantum
field name (`ethics`, `physics`, `mathematics` — never plurals) from an ordinary
plural that happens to end in `-ics` (`critics` 254, `topics` 164,
`characteristics` 142, `logics` 116, `tactics` 22, `heuristics` 10). A rule
keyed on the `-ics` ending would wrongly split all of the latter. The conf file
therefore lists the field names explicitly and deliberately leaves the plurals
alone. 47 lemmas matched the shape; 21 were real errors.

**The table is keyed on surface form, not on the wrong lemma.** Remapping the
lemma `aesthetic` → `aesthetics` would relabel the 330 genuine adjective uses.
Keying on the surface splits `aesthetics` (field, 206) from `aesthetic`
(adjective, 330), which is the correct outcome.

## Error 2 — back-formed bases that are not English words

Nine lemmas in the vocabulary were forms spaCy invented and that never occur as
a surface form anywhere in the corpus:

| surface(s) | spaCy lemma | corrected |
|---|---|---|
| `species` (207) | `specie` | `species` |
| `senses` (242) | `sens` | `sense` |
| `turing` (421) | `ture` | `turing` |
| `distinguishes` (36) | `distinguishe` | `distinguish` |
| `gendered` (37) | `gendere` | `gender` |
| `identifies` (20) | `identifie` | `identify` |
| `hoped`/`hopes`/`hoping` (54) | `hop` | `hope` |
| `taxes` (20) | `taxis` | `tax` |
| `logos` (36) | `logo` | `logos` |

Detection rule: a lemma above `min_freq` that never appears as its own surface
form. That flags 43 lemmas; 9 were errors and 34 were correct lemmas of verbs
whose base form simply never occurs (`make` from `makes`/`making`, `say`,
`take`, `being` from `beings`). After the fix, only the 34 correct ones remain.

### Verifying these are artifacts, not corpus vocabulary

`specie` and `sens` are both real English/French words, so "spaCy invented it"
needed checking against the raw article text rather than assumed. Counting
whole-word matches over the 182 fetched SEP articles, before any parsing:

| form | occurrences in raw SEP text | verdict |
|---|---|---|
| `specie` | **0** | pure artifact of `species` (260) |
| `sens` | **2** | both French; see below |
| `ture` | **0** | pure artifact of `turing` (586) |
| `logo` | **0** | pure artifact of `logos` (37) |

The two `sens` hits are a French title in the *Antoine Arnauld* entry
("…du bon sens, written the year before his death…"). They are not the source of
the 242-occurrence `sens` vocabulary entry, which came entirely from the surface
form `senses` (249). Independently, spaCy lemmatizes a genuine `sens` token to
**`sen`**, not `sens`, so those two occurrences never landed in that bucket — and
at n=2 they are far below `min_freq=20` anyway.

Confirming spaCy is the source, directly:

```
"Several species evolved together."      species -> specie   (NOUN)
"The species is endangered."             species -> species  (NOUN)
"He distinguishes two senses of X."      senses  -> sens     (NOUN)
```

Note `species` is lemmatized inconsistently — correct when the tagger reads it as
singular, back-formed when it reads plural. That is why 207 of 260 occurrences
were affected rather than all of them.

**Why keying on the surface form makes this safe regardless.** The table maps
`senses -> sense`, not `sens -> sense`. A genuine `sens` or `specie` token in the
corpus is therefore never rewritten — the fix can only ever fire on the surface
forms named, so it cannot damage a real word that happens to look like one of
spaCy's mistakes. This is the same property that lets `aesthetics -> aesthetics`
coexist with the untouched adjective `aesthetic`.

`turing` is the largest single item at 421 occurrences — worth deciding whether
it belongs in the vocabulary at all, since it is a proper noun that escaped the
`PROPN` filter. Left in for now; add to `stopwords/english.conf` to drop it.

## What is *not* fixed here

Participial adjectives (`devoted`, `alleged`, `determined`, `compelling`) keep
their own inflected lemma rather than collapsing to the base verb. This looked
like a third error class but is not one: spaCy tags them plain `JJ`, not
`VBN`/`VBG`, so there is no signal distinguishing a lemmatizer failure
(`devoted` → `devote`) from a genuinely lexicalized adjective (`determined`,
`concerned`), and the same shape also covers deverbal nouns that must not
collapse (`building`, `beginning`, `accounting`, `bearing`). 104 vocabulary
entries have the shape. They are handled instead as gated merge candidates in
[[derivational-variant-merging]], where embedding cosine is available as the
discriminator.

## Reproducing

`scripts/tools/` has no permanent audit script; the one used here lived in the
session scratchpad. It parses the corpus via `corpus.build.build_english_corpus`
+ `corpus.parse.parse_sep_article`, tallies `lemma -> Counter(surface)` over
tokens that pass the SEP content filter, and applies the two detection rules
above.
