## Chinese segmentation

There are advantages and disadvantages to performing word segmentation on the Chinese side of the text. The Kyoto treebank treats all characters as independent units of meaning, except for some names (e.g. 孟子) and numbers (e.g. 三十三, thirty-three). Indeed, Classical Chinese is largely monosyllabic, and treating characters as atomic units of meaning has some advantages, including keeping the pipeline simple. Yet there are synonym and idiomatic compounds in the text that this approach obscures, including the important Confucian term 君子 (gentleman).

## 6A

The single-term approach can also create readability problems. Without segmentation, the co-occurrence graph for 仁 and 義 in Mengzi 6A (告子上) is dominated by a cluster of terms with a high PMI relation between them: 杞, 柳, 桮, 棬, 戕, and 賊. This cluster is misleading without context. 杞 and 柳 both mean "willow tree;" they have a high co-occurrence value because they are always paired together in this chapter to express this meaning. Likewise, 戕賊 forms a synonym compound meaning "to harm/injure." The line between 猶桮 literally means "cups and bowls," though the 

杞柳, 桮棬, and 戕賊

The high relations between these compound components edge out the high relations between the target and other close terms, such as nature 性 (or human nature 人性, as it appears in the multi-character vocabulary).

....

**Note:** GujiRoBERTa only tokenizes single characters. Multicharacter embeddings are the mean of the term's single character embeddings. This process ([[Pooling|mean pooling]]) is simple but has good empirical support (citation?). That said, it could be considered an advantage that single-character tokens stick to the GujiRoBERTa embeddings more closely.