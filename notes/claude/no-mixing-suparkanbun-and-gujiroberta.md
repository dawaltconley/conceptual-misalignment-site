Another question: can I use the GujiRoBERTa model in the parsing / tokenization
phase of either suparkanbun or cltk (probably using their ollama integration)?
Would that streamline the embedding generation?

A few things are getting conflated here, so let me untangle them — the short
version is no, and you don't want to.

GujiRoBERTa is an encoder, not a parser

GujiRoBERTa is a masked-language-model encoder — it turns characters into
contextual vectors. That's it. It has no segmentation/POS/dependency capability
on its own. suparkanbun and CLTK do parsing with task heads trained on top of a
specific base model — suparkanbun's SuPar biaffine heads are trained against
roberta-classical-chinese; CLTK's lzh pipeline is Stanza's trained models. The
parser is the head, and a head only works with the representations of the base
model it was trained on. So you can't drop GujiRoBERTa into suparkanbun's
tokenization phase and get a parse back — you'd have to train new parsing heads
on GujiRoBERTa against a treebank. Possible, but it's an ML project, not a
swap.

ollama doesn't apply

ollama serves generative LLMs (GGUF: Qwen, Llama, …) and exposes
generate/chat/embeddings APIs. GujiRoBERTa is a bidirectional masked encoder
from HuggingFace transformers — it isn't a generative model and isn't something
ollama runs or that you'd route a parser through. (What you did earlier with
XunziALLM was a generative LLM doing segmentation by prompting, via llama.cpp —
a completely different mechanism. Neither suparkanbun nor CLTK uses ollama for
classical-Chinese tokenization.)

The deeper point: segmentation and embedding are already decoupled

This is the reason it wouldn't streamline anything. For your embedding
pipeline:

- Embedding needs no segmentation. GujiRoBERTa is character-tokenized — you
  feed it the raw passage text and it embeds every character in context.
  Segmentation isn't an input to that.
- Segmentation only picks node granularity — it decides which character-spans
  (仁, or 孟子 as one node vs two) become vocabulary nodes, i.e. which spans
  you mean-pool over after the embedding.

So there's no shared work to fold together: the parser decides spans,
GujiRoBERTa embeds characters, and they never need to be the same model. Your
.conllu (or suparkanbun later) supplies the spans for free; GujiRoBERTa
supplies the vectors. Two cheap, independent steps.

The one real "one model" path (and why it's not worth it now)

The only way to genuinely share a model would be to fine-tune a lightweight
token-classification (BIES segmentation) head on GujiRoBERTa — and you happen
to have labels for it (your Xunzi output / the .conllu). Then a single forward
pass would yield both segmentation tags and the last_hidden_state you embed
from. That's a legitimate design, but:

- It's a training task, not configuration.
- The payoff is tiny — the embedding pass is already cheap (your notes flag the
  segmentation as the expensive part, which the .conllu/suparkanbun already
  solves). You'd be saving one inexpensive encoder pass.

I'd only pursue it if you specifically wanted a self-contained "GujiRoBERTa
segments-and-embeds" artifact for the dissertation's methods story.

Recommendation

Keep them decoupled, which is also the simplest:

1. Segmentation/POS → your .conllu now (via conllu_to_docs), suparkanbun later.
   Emits spaCy Docs with spans.
2. Embeddings → GujiRoBERTa over the raw passage text, extract the span vectors
   (exactly what model.embed already does).

Don't try to route GujiRoBERTa through suparkanbun/CLTK/ollama — it can't
parse, and there's no efficiency to reclaim because the two jobs were never
coupled in the first place.
