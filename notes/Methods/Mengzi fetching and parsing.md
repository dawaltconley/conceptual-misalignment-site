A full, digitized text of the Mengzi is freely available via the Chinese Text Project (ctext.org).[^1] I planned to download this text using ctext's public API and segment it using the [[Xunzi ALLM]], following Wu & Wang (2025). [[Xunzi segmentation|This proved challenging.]]

I considered several options available for tokenizing and POS-tagging Classical Chinese texts
### Mengzi

1. Originally: ctext.org API
2. Then local .txt files, after ctext closed full-text API
3. Then local .conllu files, from the [Classical Chinese Universal Dependencies Treebank](https://github.com/UniversalDependencies/UD_Classical_Chinese-Kyoto.git)
[[SEP fetching and preprocessing]]

[^1]: Ctext.org seems to have closed this API sometime during the development of this project. I already had the text downloaded locally and was able to continue using that. But this partially contributed to my decision to switch to the Kyoto treebank.
