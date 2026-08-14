1. Kyoto UD treebank
2. Segmentation
	1. Rule-based from UD treebank
	2. Statistical from [[Xunzi ALLM]]
	3. In both cases, the merged multi-character term inherits its POS from one of the terms in the UD treebank
		1. For Xunzi merges, this POS may be arbitrary
		2. Since POS is only used to exclude certain terms from the corpus, we apply the rule: only merge terms if the resulting POS does *not* result in the merged word being excluded from the corpus.
		3. In practice: merged Xunzi terms all consist in parts that match the whitelisted POS

See [[tagger-comparison]] for a discussion of the relative merits of different approaches to Classical Chinese segmentation and POS-tagging. Ultimately these were abandoned in favor of parsing the Kyoto treebank.