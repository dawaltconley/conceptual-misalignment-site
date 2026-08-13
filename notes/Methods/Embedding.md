## Outline

1. Build vocab with minimum corpus frequency + targets
2. Segment at sentence and document (chapter/article) boundaries and greedily pack these under the 512-token threshold for the RoBERTa models
	1. Maybe see [[full-context-embedding]]
3. Encode: tokenize and generate embeddings for each token
	1. Remap these tokens and their embeddings onto the parsed spaCy tokens
		1. See [[why-not-a-multilingual-model]] for the rational behind using separate RoBERTa models for each language.
	2. **mean pool** subword tokens that form parts of a single spaCy token (this includes every multicharacter Chinese term
	3. See [[no-mixing-suparkanbun-and-gujiroberta]] on the rational for retokenizing using the transformer model
4. Pool: **max pool** different occurrences of the same token into a single cross-corpus vector
	1. See [[subword-pooling-methodology]] for a discussion on the difference between the subword and cross-document pooling methods.
## Methods

