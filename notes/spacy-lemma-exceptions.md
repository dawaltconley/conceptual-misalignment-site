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

## Error 3 — a lemma error can erase an entire rendering

Errors 1 and 2 damage the *vocabulary*: a node lands in the wrong bucket, and
you see it as scatter clutter. The same mistake landing on a
`config.TERMS` rendering is worse, because a `Rendering` is matched against a
token's **lemma** (`models.Rendering.matches`), so a bad lemma does not
mis-bucket the term — it deletes it from the run.

禮's `mores` was dead this way. `en_core_web_sm` reads it as a plural and emits
the lemma `more`:

| surface | spaCy lemma | POS | occurrences |
|---|---|---|---|
| `mores` | `more` | NOUN | 11 across 7 articles (montesquieu 1, weber 2, diderot 3, godwin 1, solidarity 2, double-consciousness 1, addams-jane 1) |

`mores` is a Latin pluralium tantum, so this is the same class as error 1 —
but the consequence is different in kind. The pattern `'mores'` never fired,
`TermData.occurrences` came out 0 for every article, `build_cooccurrence_network`
returned `None`, and all 15 `public/sep/mores_*.json` files were written with
`"network": null`. Fixed by `mores -> mores`.

**Why it stayed hidden.** Nothing distinguishes an erased rendering from an
honest one. A term the corpus does not discuss also has 0 occurrences and also
writes null networks, and the only console output was one
`no co-occurrence for mores in …` line per article, which reads as a sparse-data
result. After the fix the same articles yield 16-node networks (`custom`,
`ethos`, `citizen`, `democracy`, `cultural`, `community`), so the data was
always there.

**This one *is* worth detecting automatically** — unlike errors 1 and 2, where
the detection rule can only propose candidates. The signature is exact: a token
whose *surface* matches a rendering's patterns while its *lemma* does not.
`renderings.check_coverage` runs it as a pipeline guard (fatal by default,
`--allow-empty-renderings` to downgrade), and
`scripts/tools/rendering_diagnostics.py` runs the full audit, which also catches
*partial* erasure — a rendering matching most of its occurrences while losing a
slice.

**But the fix still must not be automatic.** The tempting shortcut is to have
`Rendering.matches` fall back to the surface form, or to auto-generate a
`surface -> surface` exception for every literal pattern. Measured over 70
sampled SEP articles, the only rendering where that would change anything is
義's `meaning`:

| surface | spaCy lemma | POS | count | matched by lemma |
|---|---|---|---|---|
| `meaning` | `mean` | VERB | 55 | 573 |

and there the loss is *correct* — the verb ("meaning that p is true") is not the
noun 義 is rendered as. A surface fallback would fold those 55 into the target
node. No token in the sample matched one rendering by lemma and a *different*
one by surface, so the hazard is exactly this: not mislabeling, but over-capture
of a POS the rendering never wanted. Detection is automatic; the conf line stays
a human decision, which is the same discipline errors 1 and 2 established.
`Rendering.pos` (unused as of this writing) is the knob that would make
auto-preservation safe, by letting a rendering declare which reading it means.

### A second, unrelated way a rendering can match nothing

Patterns are matched against one token's lemma, so a **multi-token** pattern can
never match on its own, no matter what the lemmatizer does. Three of the
concepts in `config.TERMS` are phrases — `'social norm*'` (禮),
`'human nature'` (性), `'care for'` (愛) — and `'heart-mind*'` (心) is in the
same position without containing a space, because the English tokenizer splits
hyphens.

No exception table reaches this, since the unit being matched is the problem.
`corpus.parse.merge_phrases` fixes it upstream instead: a spaCy `Matcher` holds
every multi-token pattern, and each match is retokenized into a **single token
whose lemma is the rendering's label**. From there the phrase is an ordinary
node — it matches its own label by construction, carries the phrase as its
display `form`, and keeps exact `token.idx` offsets, because merging does not
alter `doc.text`. The embedder then sees the whole phrase as one span, which is
the right span to embed anyway.

Two details worth knowing:

- **Patterns are split by the real tokenizer, not on spaces**, so `heart-mind*`
  becomes `heart` `-` `mind*` and is caught like any phrase. This is also the
  definition `renderings.RenderingAudit.multiword` uses, so the guard classifies
  a failure the same way the merger would.
- **Each pattern is registered twice, over `LEMMA` and over `LOWER`.** A phrase
  inflects on its head (`social norms` lemmatizes to `social norm`), and either
  spelling should reach `social norm*`. Matching on either attribute is strictly
  more forgiving than the single-token rule, never less.
- Merged attributes other than the lemma are **inherited from the span's
  syntactic head**, so `social norms` stays NOUN and `care for` stays VERB
  rather than taking a POS hardcoded by the merger.

Measured on four SEP articles, `social norm*` went from 0 occurrences and a null
network to 7 occurrences and a 16-node network (`disgust`, `conformity`,
`circumstance`, `capacity`), with `social norms` as the display form.

A match that would cross a sentence boundary is skipped — "That is not social.
Norms, however, are." must not merge — since `build_segments` packs whole
sentences and a token straddling the break would corrupt the packing.

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

For errors 1 and 2 there is no permanent audit script; the one used here lived in
the session scratchpad. It parses the corpus via
`corpus.build.build_english_corpus` + `corpus.parse.parse_sep_article`, tallies
`lemma -> Counter(surface)` over tokens that pass the SEP content filter, and
applies the two detection rules above.

Error 3 does have one, since its rule is exact:

```bash
scripts/.venv/bin/python scripts/tools/rendering_diagnostics.py --per-term 12
```

It writes `analysis/sep/rendering_diagnostics.md` — every rendering with its
matched count, its shadowed surfaces (`surface -> lemma (POS)`), and the ones
that matched nothing at all.
