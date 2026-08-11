## Me

I have a question regarding the earlier plan about aligning the two semantic
spaces.

What are the downsides to using a single, multilingual BERT / Roberta model to calculate embeddings for both corpora? I assume it will be less accurate than the the fine-tuned models, especially for classical Chinese. But I'm wondering about having it as one output among others.

## Claude

Your instinct about accuracy is right, but it's the _third_-most important
problem. The two bigger ones are worth being explicit about, because one of
them cuts at your thesis directly.

**The circularity problem.** Multilingual encoders acquire their cross-lingual
alignment from parallel or comparable corpora — that's the training signal that
makes "dog" and "chien" land near each other. So the model arrives with a
translation prior already baked in: it has effectively been taught that 仁 ≈
benevolence, because that's the equivalence its training data asserts. If you
then measure the distance between 仁 and "benevolence" in that space, you're
substantially measuring _the model's inherited translation convention_, not the
conceptual structure of the Mengzi versus the SEP. A small residual distance
would be weak evidence of alignment, since the model was optimized to produce
exactly that. This is why I'd resist making a multilingual model your primary
instrument for a _misalignment_ claim: the thing you want to detect is the
thing it was trained to suppress. Worth noting this gets worse, not better,
with translation-tuned sentence encoders like LaBSE — those are explicitly
bitext-mining models, so their alignment prior is even stronger.

**Language-identity dominance.** In practice, representations from multilingual
models cluster first by language and only secondarily by meaning — a large
chunk of the variance is a "which language is this" subspace. Raw cross-lingual
cosine similarities are therefore compressed and confounded: everything Chinese
looks similar to everything else Chinese. The standard mitigation is
per-language mean-centering (subtract each language's centroid before
comparing), and if you run this variant you should do that and say so. But
notice what centering is: an alignment step. So the apparent advantage — "no
alignment needed" — partly dissolves; you've just moved the alignment inside
the model and made it less inspectable.

**Then the accuracy issue you anticipated.** mBERT/XLM-R saw modern Chinese web
and Wikipedia text, not pre-Qin philosophical prose. Expect worse tokenization
of classical constructions, and expect the model to read 仁, 義, 禮 through
their modern-Mandarin senses rather than their Warring-States ones — which is
precisely the semantic drift your dissertation is about. GujiRoBERTa exists to
avoid that. There's also a script wrinkle: your corpus is traditional, and
multilingual models' Chinese exposure skews simplified.

Given all that, **yes, run it — but frame it correctly.** Its value isn't as a
competing measurement; it's as a _contrast case with a known bias_, and the
direction of that bias is predictable. If your Procrustes-aligned monolingual
spaces show substantial misalignment for 仁 while the multilingual model shows
little, that difference is interpretable rather than embarrassing: it's the gap
between what the texts support and what standard translation practice (encoded
in the training data) assumes. That's a genuinely nice argumentative move — the
multilingual model becomes a proxy for the received translation convention
you're critiquing.

Two practical notes if you do include it. Report it as a
robustness/triangulation panel with the circularity caveat stated once,
plainly, rather than as one of two equal results — otherwise a reader will ask
why two methods disagree and you'll have lost control of the framing. And for
the MVP, I'd put it _after_ the primary pipeline works end to end; it's cheap
to add later (one model swap, no new analysis code), whereas the Procrustes
path is where the actual methodological risk lives. If Aug 15 gets tight, this
is the piece to defer.
