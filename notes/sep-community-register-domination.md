# SEP communities are register-partitioned, not conceptual

Diagnosis of the SEP embedding communities (run: `resolution=2.0`, PROPN dropped, bibliography
sections stripped from `#main-text`, Latin-abbreviation stopwords, `min_freq=20`, no parser/ner).
Snapshot: **6148 nodes, 21 communities, 38 targets**.

## What improved over the res=1 baseline
- The single 1760-node hub is gone — split into 21 communities.
- The bibliography strip + PROPN drop dismantled the old pure author/citation community
  (was ~330 nodes: `sidgwick parfit passim vv`), and the Latin abbreviations are gone.

## The core problem (unchanged): clustering axis is grammatical register, not meaning
Every community is a **part-of-speech + suffix-morphology + register** band, e.g.:

| community | what it is |
|---|---|
| C6  | Latinate `-ate/-ive` verbs (`concern provide require determine constitute`) |
| C13 | vivid verbs (`blur marshal transgress undercut bypass`) |
| C8  | relation verbs (`associate accord apply contrast relate`) |
| C1 (13 targets) | **light verbs + common words** (`work time way find go give mean bear`) |
| C10 (13 targets) | **abstract nouns** (`character function property quality reason process term`) |
| C14 (4 targets) | domain adjectives (`social ethical mental psychological cognitive`) |
| C18 (2 targets) | generic adjectives (`rudimentary forceful innocuous stringent refined`) |
| C2 / C7 | `-ible` / `-ive` adjectives |
| C19 / C12 / C5 | `-ization` nominals / `-ism` positions / `-ism` jargon |

**Targets fragment by POS, not concept.** The three renderings of 仁 scatter across three
communities purely by word class: `benevolence`→C14 (adj), `humaneness`→C18 (adj),
`humanity`→C15 (noun). ~26 of 38 targets sit in just the two generic hubs (C1 verbs, C10 nouns),
so a target's community tells you its part of speech, not its meaning.

**Meaning is present but swamped:** C15 is a genuinely coherent cluster —
`humanity/kin/parents` with `people family person culture citizen`. The exception that proves
the rule.

## Junk pockets still present
- **C20 (21): single letters** `x v m c p d t s e h w …` (enumeration/variable junk) — filtered
  out manually rather than in the pipeline.
- **C16 (203): author-derived adjectives** `gricean millian popperian quinean fregean rawlsian
  lockean` + leaked surnames `sidgwick sartre leibniz gettier` + foreign tokens
  `csmk brsūbh tamhīdāt`. Dropping PROPN missed these because `-ian/-ean/-ist` author-forms tag
  as **ADJ**, not PROPN. (See the NER-filter sketch below and the curated list.)
- **C11 (29): document-type nouns** `anthology monograph pamphlet preface reprint catalogue`
  — residual bibliography vocabulary that occurs in body text.
- **C3 (13): ordinals** `sixteenth seventeenth …` — century references.

## Recommendation
Whole-vocabulary Louvain over these embeddings will always be register-dominated — resolution
and stopwords only thin it, they don't change the clustering axis. The structural fix is to
**stop clustering the full 6148-word lexicon** and instead detect communities only within the
**union of the target neighborhoods** (the virtues + their nearest neighbors, which we already
compute per target in `sep/*_embeds.json`). That is both the research question and immune to the
register hub.

Quick wins that keep the global view: (1) uncomment the generic-noun/light-verb `ENGLISH_STOPWORDS`
(thins C1/C10), (2) single letters (C20) filtered manually, (3) author-adjective filter for C16.

## Sketch: NER-based author-adjective filter
NER **alone won't** catch the adjectives — `en_core_web_sm` tags `kantian/rawlsian` as ADJ, not
`PERSON`. So combine NER (to harvest names) with a derivational match (to catch the adjectives):

```python
# 1. Harvest a person gazetteer from the corpus (needs ner enabled during parse,
#    or a one-off pass; run on TEXT, not isolated vocab tokens — NER needs context).
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser"])   # keep ner
names: set[str] = set()
for doc in nlp.pipe(article_texts, batch_size=32):
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            for tok in ent:
                if tok.is_alpha and len(tok) > 2:
                    names.add(tok.lemma_.lower())          # 'kant', 'sidgwick', 'rawls'

# 2. Drop a vocab token if it *is* a harvested name, or is an adjectival/-ism
#    derivation of one (strip the suffix and check the stem against the gazetteer).
_SUFFIXES = ("ian", "ean", "esque", "ism", "ist", "ite")
def is_author_derived(lemma: str, names: set[str]) -> bool:
    if lemma in names:
        return True
    for suf in _SUFFIXES:
        if lemma.endswith(suf):
            stem = lemma[: -len(suf)].rstrip("-")
            # 'kantian'->'kant', 'russellian'->'russell'; allow a dropped final 'e'
            if stem in names or stem + "e" in names or stem.rstrip("i") in names:
                return True
    return False
```

Wire point: fold `is_author_derived` into `content_key` (occurrences.py) alongside the existing
`is_stop` / `content_pos` checks, passing the gazetteer through the pipeline.

**Caveats / why it's only a sketch:**
- Re-enables `ner` (the component was disabled for speed) — adds parse time; harvest once and
  cache the gazetteer to `data/`.
- Stem→name matching is fuzzy: `-ian` drops/reduces stems irregularly (`hume→humean`,
  `plato→platonic` uses `-ic` not listed; `hegel→hegelian` fine). Expect misses and the odd
  false positive (`utilitarian` stem `utilitar`, `christian` stem `christ` — the latter *is* a
  name; whitelist common non-author `-ian/-ism` words).
- Foreign transliterations (`csmk brsūbh tamhīdāt`) aren't person-derived; a
  non-ASCII / not-in-English-lexicon filter is a separate, simpler rule.
- Cheaper alternative to NER: a static gazetteer of philosopher surnames (there are curated
  lists) + the same suffix derivation — no `ner`, no parse cost, but needs a maintained list.
