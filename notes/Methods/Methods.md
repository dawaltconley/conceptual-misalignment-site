1. Target term definition
	1. Using wordstems to merge core term variants (and to search for them in the SEP)
2. Corpus fetching
	1. [[Mengzi fetching and parsing]]
	2. [[SEP fetching and preprocessing]]
3. Parsing
	1. Mengzi
		1. Kyoto UD treebank
		2. Segmentation
			1. Rule-based from UD treebank
			2. Statistical from Xunzi ALLM
			3. In both cases, the merged multicharacter term gets its 
	2. SEP
	3. Both parsed into the same data structure using spaCy
4. Filtering
	1. Stopwords
	2. Lemmatization
	3. POS exclusion (all but nouns, verbs, adjectives (adverbs?)
5. Embedding
	1. Build vocab with minimum corpus frequency + targets
	2. Segment at sentence and document (chapter/article) boundaries and greedily pack these under the 512-token threshold for the RoBERTa models
	3. Encode: tokenize and generate embeddings for each token
		1. Remap these tokens and their embeddings onto the parsed spaCy tokens
		2. **mean pool** subword tokens that form parts of a single spaCy token (this includes every multicharacter Chinese term
		3. See [[no-mixing-suparkanbun-and-gujiroberta]] on the rational for retokenizing using the transformer model
	4. Pool: **max pool** different occurrences of the same token into a single cross-corpus vector
6. ...




## Visualization

To give a bird's-eye view of semantic space of each corpus, the core vocabulary is represented in a pair of scatter plot graphs. I tried several 

The 

See [[community-legend-ordering]]