### SEP

I used the *Stanford Encyclopedia of Philosophy* (SEP) for my English-language corpus. The SEP is a public, peer-reviewed encyclopedia, with nearly 1800 articles covering thinkers and topics across all areas of philosophy. It can thus serve as a fairly representative sample of academic writing in contemporary Anglophone philosophy.

The size and breadth of the SEP also means, however, that it contains an incredibly large and diverse vocabulary, which includes technical terms from nearly all fields of philosophy. This can make identifying semantic patterns unwieldy: calculating semantic similarity using word embeddings will tend to 

That both my corpora are publicly available also helps remove some of the "black box behavior" that can accompany NLP algorithms (Alharbi et al. 2021). Users can quickly follow a link to the primary text and search a given term to get a sense for its context. This is especially useful when algorithms like PMI sometimes surface unhelpful information. For example, some co-occurrence pick up terms from [[Chinese segmentation|running analogies in the Mengzi]] or [[thought experiments in the SEP]]. 

This project prioritized the most relevant articles in the SEP for each term of interest. I 