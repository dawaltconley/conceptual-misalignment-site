# Full-context embedding: what the model reads vs. what we keep

A note on a design point that looks wrong at a glance: the occurrence parsers filter
out stopwords, punctuation, and non-content words. That filtering does **not** remove
those tokens from the text the model embeds — it only chooses which tokens become
nodes/occurrences. The contextual embeddings are computed over complete sentences.

Reference: commit `6e6bd8c` (`scripts/embeddings/{occurrences,sep_occurrences}.py`,
`scripts/embeddings/model.py`).

## The principle

A contextual model (roberta-base for English, GujiRoBERTa for Chinese) assigns each
token a vector *as a function of its whole surrounding sentence*. The value of the
method is that `benevolence` in "a benevolent ruler cares for the people" gets a
different, context-shaped vector than `benevolence` in a definition of justice. If we
stripped stopwords before feeding the model, we would destroy exactly that context and
get degraded, unnatural embeddings. So the text handed to the model is the **full,
verbatim sentence**; the stopword/POS filter runs *after* the model, selecting which
words we extract a vector for.

## How the code enforces this

Two separate things happen in the parser (`sep_occurrences.parse_docs`, and the Chinese
analogue in `occurrences.build_segments`):

1. **The sentence text is kept whole.** `ParsedSentence.text = sent.text` — the entire
   sentence, stopwords and punctuation included. The token loop's `continue` on
   non-content tokens only skips adding them to the `toks` list (the spans of interest);
   it never edits the text.
2. **`toks` are spans, not a filtered string.** Each kept token is stored as
   `(key, start, end)` — a character range *into the full sentence text*.

`build_segments` then packs the **full** sentence text into each `Segment`
(`cur.text += sep + s.text`) and offsets the spans into the packed text. So a
`Segment.text` is real, uninterrupted prose.

Finally `model._embed_batch` tokenizes `[r.text for r in batch]` — the full segment
text — runs the encoder over it, and only *afterward* pulls out each occurrence:

```python
hidden = self.model(**enc).last_hidden_state          # embeddings from full context
for span in rec.spans:
    tok_idx = [j for j, (a, b) in enumerate(offs)
               if b > a and a < span.end and b > span.start]   # subwords overlapping the span
    vec = hidden[i, tok_idx].mean(axis=0)             # mean over the word's subwords
```

So a word's vector is the model's output *with the whole sentence attending over it*,
then averaged over that word's subwords. Stopwords are part of the attention context;
they are simply never promoted to graph nodes (you do not want `the`/`of`/`之`/`也` as
points in the semantic space). Downstream steps — max-pool across occurrences, and later
mean-centering + L2 normalization — operate on those already-full-context vectors; there
is nothing to "normalize afterward" to compensate for stripping, because nothing was
stripped from the model input.

## The one real boundary: context = sentence(s), not the whole document

"Full context" means the sentence(s) packed into a segment, not the entire article or
Mengzi book. Sentences are packed greedily up to the model's 512-token cap and a segment
never crosses a source unit (a SEP article on the English side, a Mengzi passage on the
Chinese side). Occurrences in the same packed segment share context; the model never
attends across a boundary the packer didn't cross. That is a deliberate, reasonable
scope (it keeps each input coherent and within the token limit), but it is the actual
limit on "context" — stopword filtering is not. Widening context would mean widening the
packing unit, not keeping stopwords.
