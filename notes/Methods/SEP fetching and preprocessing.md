I used the *Stanford Encyclopedia of Philosophy* (SEP) for my English-language corpus. The SEP is a public, peer-reviewed encyclopedia, with nearly 1800 articles covering thinkers and topics across all areas of philosophy. It can thus serve as a fairly representative sample of academic writing in contemporary Anglophone philosophy.

This project prioritized the most relevant articles in the SEP for each term of interest, by using the SEP's search page to identify the most relevant articles for a given word stem.[^2]

~~It also filtered out any articles concerned primarily with Chinese philosophy. If included, the most common translations for core Chinese philosophical terms, such as "benevolence" and "righteousness," would show convergence, even though there is no significant convergence between these terms in Anglophone philosophy, as the current results show.~~

A web scraper pulled the first {eight} articles after filtering out any that were primarily concerned with Chinese philosophy. It identified these by querying an undocumented API hosted by the Internet Philosophy Ontology Project (InPhO) at [hypershelf.org](https://www.hypershelf.org/), which returns the results of a topic model trained on the SEP. Any article which passes a threshold (25%) of topic 78 on InPhO's [100 topic model](https://www.hypershelf.org/sep/100/?doc=) is discarded. This is observed to reliably discard articles concerned with Chinese philosophy. ~~Pruning Chinese philosophy articles from the SEP corpus ensures that the most common translations for core terms, like "benevolence" and "righteousness," do not show artificial convergence.~~ If included, the most common translations for core Chinese philosophical terms, such as "benevolence" and "righteousness," would show convergence, even though there is no significant convergence between these terms in Anglophone philosophy, as the current results show.

After recording the number of times each core term occurs in Chinese philosophy articles, they are discarded. Articles with fewer than {3} occurrences of a target term are also discarded. The body text of the rest is scraped and cleaned, removing references and inline citations, to avoid polluting the results.[^3]

**Maybe here:** discuss [[Merging derivatives]] 

## Supporting reasons
### Large vocab

Why didn't I just pull the entire SEP? After all, 1800 articles isn't too much for the kinds of language processing I'm doing.

The size and breadth of the SEP also means, however, that it contains an incredibly large and diverse vocabulary, which includes technical terms from nearly all fields of philosophy. This can make identifying semantic patterns unwieldy: calculating semantic similarity using word embeddings will tend towards anisotropy in extremely large or diverse corpora.[^1]
### Transparency

That both my corpora are publicly available also helps remove some of the "black box behavior" that can accompany NLP algorithms (Alharbi et al. 2021). Users can quickly follow a link to the primary text and search a given term to get a sense for its context. This is especially useful when algorithms like PMI sometimes surface unhelpful information. For example, some co-occurrence pick up terms from [[Chinese segmentation|running analogies in the Mengzi]] or [[thought experiments in the SEP]]. 



[^1]: Source?

[^2]: The SEP search natively supports glob searches for partial matches on a word, so this was straightforward to implement.

[^3]: Example?
