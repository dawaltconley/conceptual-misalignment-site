This is all very good to know. I have many questions, which I'm going to throw
at you. You don't have to give equal attention to each; only respond in detail
to those you think would be most informative.

> Embeddings capture paradigmatic, not syntagmatic, similarity.

Is register-similarity the same as paradigmatic similarity? It seems like
etymology and suffixes dominate here, I would have expected other areas.

> RoBERTa tokenizes canaliz-ation, symboliz-ation, relativiz-ation into shared
> subword pieces; your subword-mean pooling then drags their vectors together.
> Your -ization, -ism, -ible communities are partly a tokenization artifact, not
> semantics.

I'd be interested in turning off English subword pooling temporarily; that was
included because GujiRoBERTa only has a vocab of single-character tokens, and
rare multichar tokens would have gotten lost. But in English those are common,
and recognizing the same root term across subword variations (e.g. symbol,
symbolic, symbolize) might help isolate it from other terms with similar
suffixes.

> Mean-centering removes one common direction, but frequency/register structure
> lives in several more. And max-pool across occurrences makes it worse than
> mean-pool would...

Please implement a Pipeline config variable to control subword pooling
(mean/max/none), and while you're at it, add another variable to control
cross-occurrence pooling (mean/max/none). These should be separate commits.
Give them their current defaults and don't update config.py yet.

Regarding the difference between topic and structure (your response to question
three), I might explore reintroducing topic (not now, at a later point). Is
there any benefit to calculating topic dynamically from the current corpus, or
could I reuse the InPhO topic API I'm currently using to exclude Chinese
philosophy articles?

Can you provide a source on why co-occurrence relates to topic?

Re a "small-but-unusually-diverse" corpus: I know you said the Mengzi was too
small to get meaningful embeddings on a per-chapter basis. But I can pull
_more_ SEP articles for each search term and generate unique embeddings by
search (I assume by-article is still too small?). Would this help? Presumably
articles in a given search would be less diverse than those across all
searches. This is a bit of a lift to implement but I'd consider it.

Good note on HDBSCAN. Make a note to try adding as an option. And make a note
to try both debiasing approaches (on a per-pipeline basis). But for the moment,
try to answer these questions. Then, summarize this conversation in a note,
with a reference list of references you've verified via internet search.
