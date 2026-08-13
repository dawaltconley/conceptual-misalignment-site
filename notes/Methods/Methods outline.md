1. Target term definition
	1. Using wordstems to merge core term variants (and to search for them in the SEP)
2. Corpus fetching
	1. [[Mengzi fetching and parsing]]
	2. [[SEP fetching and preprocessing]]
3. Parsing
	1. [[Mengzi parsing|Mengzi]]
		1. Kyoto UD treebank
		2. Segmentation
			1. Rule-based from UD treebank
			2. Statistical from [[Xunzi ALLM]]
			3. In both cases, the merged multi-character term inherits its POS from one of the terms in the UD treebank
				1. For Xunzi merges, this POS may be arbitrary
				2. Since POS is only used to exclude certain terms from the corpus, we apply the rule: only merge terms if the resulting POS does *not* result in the merged word being excluded from the corpus.
				3. In practice: merged Xunzi terms all consist in parts that match the whitelisted POS
	2. [[SEP parsing|SEP]]
	3. Both parsed into the same data structure using spaCy
4. [[Filtering]]
	1. Stopwords
	2. Lemmatization
	3. POS exclusion (all but nouns, verbs, adjectives (adverbs?)
5. [[Embedding]]
	1. Build vocab with minimum corpus frequency + targets
	2. Segment at sentence and document (chapter/article) boundaries and greedily pack these under the 512-token threshold for the RoBERTa models
		1. Maybe see [[full-context-embedding]]
	3. Encode: tokenize and generate embeddings for each token
		1. Remap these tokens and their embeddings onto the parsed spaCy tokens
		2. **mean pool** subword tokens that form parts of a single spaCy token (this includes every multicharacter Chinese term
		3. See [[no-mixing-suparkanbun-and-gujiroberta]] on the rational for retokenizing using the transformer model
	4. Pool: **max pool** different occurrences of the same token into a single cross-corpus vector
		1. See [[subword-pooling-methodology]] for a discussion on the difference between the subword and cross-document pooling methods.
6. Debiasing
	1. Centering - subtract centroid from each vector
	2. [[ABTT Debiasing|"All-but-the-top" remove top-principle components]]
7. Filtering 2: merge English derivatives
	1. Once debiasing was accomplished these started clumping together in the similarity graph + embedding space. This was a sign that things were working. But it also visually cluttered both graphs.
	2. See [[derivational-variant-merging]]
8. Network calculations \[maybe discuss this all under "Visualizations"]
	1. Similarity
	2. Co-occurrence / PMI
	3. Louvain communities
	4. Exported to static JSON files for consumption by the browser
9. Visualization
	1. [[Network diagrams]]
		1. What means what: stronger edges = closer nodes and thicker lines
		2. ...
	2. [[Scatter plots]]
		1. What means what: color, opacity, size
		2. PCA vs t-SNE