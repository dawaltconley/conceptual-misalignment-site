# Why embedding communities cluster by register, not concept — and what to do

Conversation summary (2026-08-01). Companion to [[register-dominant-clustering]] /
`sep-community-register-domination.md`, which hold the concrete SEP snapshot. This note is the
conceptual + methodological digest, with a verified reference list at the bottom.

## The phenomenon
We pool contextual embeddings to **one type-level vector per word**, then build a cosine
kNN/quantile graph and run Louvain. The resulting communities partition by **part-of-speech +
suffix morphology + frequency/register**, with topic only a weak secondary signal. Why:

- **LMs encode POS/syntax very strongly and linearly.** Syntactic role is among the most
  decodable properties of BERT/RoBERTa representations (Tenney et al. 2019). A pooled type vector
  therefore carries a strong "which syntactic slots do I appear in" signature → light-verb,
  abstract-noun, adjective communities.
- **Embeddings capture *paradigmatic* (substitutability) similarity, not *syntagmatic* (topical
  co-presence) relatedness.** Substitutable words share POS by construction, so embedding-cosine
  clustering is inherently a same-type grouping (Sahlgren 2008; and see the co-occurrence note
  below).
- **BPE subwords pull morphological families together.** Shared subword pieces (`-ization`,
  `-ism`, `-ible`) drag those words' vectors together under subword-mean pooling — a tokenization
  artifact, not semantics.
- **Anisotropy + frequency.** Contextual spaces are anisotropic and frequency is encoded in
  position (Ethayarajh 2019; Gao et al. 2019). Mean-centering removes *one* common direction;
  register/frequency lives in several more. **Max-pool across occurrences** worsens it vs.
  mean-pool: the per-dim max is driven by outlier contexts and form-encoding dims.

## Is it common? Is our corpus large/diverse?
Common, and the lever is **scope/diversity, not size** — but that is a claim about *what the
clustering finds*, not about anisotropy (caveat below). Clustering the *whole* vocabulary makes
POS/register the only axis separating *every* word pair, because topical structure is local and
sparse: nearest-neighbour sets are largely POS-homogeneous, enough that "proportion of nearest
neighbours sharing a POS" is used as a model diagnostic (Hershcovich et al. 2019), and even
*within* local clusters the dominant directions encode structural information (punctuation/stop
words) or tense rather than sense (Rajaee & Pilehvar 2021). The diversity half has direct support
from retrieval: a global space over a topically diverse corpus "risks capturing only coarse
representations of those topics dominant in the corpus," while a topic-restricted space recovers
topical neighbours for the same word — global neighbours of *cut* are cutting/squeeze/slash,
topic-local ones tax/deficit/budget (Diaz et al. 2016). Our SEP corpus is **small but diverse**:
~373 articles / ~1–1.5M tokens (tiny for NLP) spanning all of philosophy (topically scattered,
jargon-heavy). Size acts on a different quantity — how well each pooled type vector is estimated
from its occurrences; neighbourhoods are unstable even for words at 100–200 occurrences
(Wendlandt et al. 2018), and small data degrades DSMs unevenly across frequency bands
(Sahlgren & Lenci 2016).

**Caveat (added 2026-08-14).** No source ties *vocabulary diversity* to **anisotropy**, and we
shouldn't imply one: anisotropy is a property of the pretrained model's training dynamics
(Ethayarajh 2019; Gao et al. 2019), fixed before our corpus is seen — the corpus only decides
which words and contexts get pooled inside that geometry. Diaz et al. and the corpus-domain
result that domain matters more than size (Lai et al. 2015) both concern spaces *trained* on the
corpus, so they transfer to us only by analogy, through the **scope of the vocabulary being
clustered** rather than through the geometry itself.

## Does the clustering algorithm matter?
Secondary — the **geometry** decides; no method recovers topic structure the vectors bury.
Louvain/modularity additionally has a **resolution limit** (Fortunato & Barthélemy 2007) that
merged everything into one hub at resolution 1 (raising it to 2 helped, but only re-partitions the
same register geometry). k-means/spectral behave similarly. **HDBSCAN** is the one worth trying
because it leaves low-density points *unclustered as noise* (would quarantine junk pockets) — but
the surviving clusters stay register bands. Levers are **representation** and **scope**, not the
algorithm.

## Generally-accepted routes to more-semantic clustering (cosine is the right metric)
Cosine is the standard similarity for embeddings; the *representation* is the problem, not the
metric. Established fixes:
- **De-bias the space.** All-but-the-top (Mu & Viswanath 2018): remove the top-k PCs (the
  frequency/common directions) — a stronger version of our single-direction centering.
  Whitening / BERT-whitening (Su et al. 2021) and BERT-flow (Li et al. 2020) push the space toward
  isotropy and reliably improve semantic similarity. Cheap; no re-embedding of the model needed.
- **Contrastive fine-tuning** (SimCSE, Gao et al. 2021): reshapes the space so cosine tracks
  meaning, not form. The "right" fix, but needs training.
- **Don't go type-level / whole-vocab.** Cluster occurrences (word-sense induction), restrict to
  one POS, or cluster only within target neighborhoods.
- **Use co-occurrence for topic** (below).

## Embeddings vs. co-occurrence = paradigmatic vs. syntagmatic
Our two network types capture *different relations*. Sahlgren's refined distributional hypothesis:
a model built from **co-occurrence** information encodes **syntagmatic** relations (words that
appear *together* → topical/associative), while one built from **shared neighbours** encodes
**paradigmatic** relations (words that are *substitutable* → same-type). So the "conceptual
grouping" we want is closer to the **PMI co-occurrence** side, and embeddings are best used for
*measuring virtue-to-virtue distances* (the alignment question), not for whole-vocab topical
clustering.

## What we changed this session (all in `Pipeline`, config-level, per-corpus)
- `resolution` (Louvain) — added; SEP set to 2.0.
- `subword_pooling` (mean/max/none) and `occurrence_pooling` (mean/max/none) — added as knobs to
  experiment with the two pooling artifacts above (defaults unchanged: mean / max).
- SEP vocab cleanup: PROPN dropped, bibliography sections stripped from `#main-text`, Latin
  abbreviations + author-derived adjectives + single letters + ordinals + generics in
  `stopwords/english.txt`.

## To try (per-pipeline)
- [ ] **HDBSCAN** as a `sim_network`/clustering option — density-based, quarantines junk as noise.
- [ ] **Debiasing**, both variants, as a per-`Pipeline` toggle: **all-but-the-top** (remove top-k
  PCs) and **whitening/BERT-whitening**. Try before graph construction and compare community
  coherence.
- [ ] Possibly: reintroduce **topic** (dynamic per-corpus, or reuse the InPhO topic API already
  used to exclude Chinese-philosophy articles).
- [ ] Possibly: **by-search embedding spaces** (articles within one search are less diverse than
  across all searches) — needs more articles per term; by-article is likely still too small.

## References (verified via search, 2026-08-01)
- Ethayarajh, K. (2019). *How Contextual are Contextualized Word Representations? Comparing the
  Geometry of BERT, ELMo, and GPT-2 Embeddings.* EMNLP-IJCNLP. https://aclanthology.org/D19-1006/
- Gao, J., He, D., Tan, X., Qin, T., Wang, L., Liu, T.-Y. (2019). *Representation Degeneration
  Problem in Training Natural Language Generation Models.* ICLR. https://arxiv.org/abs/1907.12009
- Tenney, I., Das, D., Pavlick, E. (2019). *BERT Rediscovers the Classical NLP Pipeline.* ACL.
  https://aclanthology.org/P19-1452/
- Sahlgren, M. (2008). *The Distributional Hypothesis.* Italian Journal of Linguistics 20(1).
  https://www.italian-journal-linguistics.com/app/uploads/2021/05/Sahlgren-1.pdf
- Mu, J., Viswanath, P. (2018). *All-but-the-Top: Simple and Effective Postprocessing for Word
  Representations.* ICLR. https://arxiv.org/abs/1702.01417
- Li, B., Zhou, H., He, J., Wang, M., Yang, Y., Li, L. (2020). *On the Sentence Embeddings from
  Pre-trained Language Models* (BERT-flow). EMNLP. https://aclanthology.org/2020.emnlp-main.733/
- Su, J., Cao, J., Liu, W., Ou, Y. (2021). *Whitening Sentence Representations for Better Semantics
  and Faster Retrieval.* arXiv:2103.15316. https://arxiv.org/abs/2103.15316
- Gao, T., Yao, X., Chen, D. (2021). *SimCSE: Simple Contrastive Learning of Sentence Embeddings.*
  EMNLP. https://aclanthology.org/2021.emnlp-main.552/
- Fortunato, S., Barthélemy, M. (2007). *Resolution Limit in Community Detection.* PNAS 104(1),
  36–41. https://www.pnas.org/doi/abs/10.1073/pnas.0605965104

Added for the scope/diversity vs. size discussion (verified via search, 2026-08-14):
- Diaz, F., Mitra, B., Craswell, N. (2016). *Query Expansion with Locally-Trained Word Embeddings.*
  ACL, 367–377. https://aclanthology.org/P16-1035/ — global embeddings over a diverse corpus
  capture "only coarse representations of those topics dominant in the corpus"; topic-restricted
  spaces recover topical neighbours (the *cut* example).
- Rajaee, S., Pilehvar, M. T. (2021). *A Cluster-based Approach for Improving Isotropy in Contextual
  Embedding Space.* ACL-IJCNLP (Short), 575–584. https://aclanthology.org/2021.acl-short.73/ —
  clusters of punctuation/stop words whose local dominant directions encode structural information,
  and verb clusters where tense dominates sense; removing those directions helps semantic tasks.
- Hershcovich, D., Toledo, A., Halfon, A., Slonim, N. (2019). *Syntactic Interchangeability in Word
  Embedding Models.* RepEval @ NAACL-HLT. https://aclanthology.org/W19-2009/ — uses shared POS among
  nearest neighbours (with POS as proxy for syntactic interchangeability) as a model diagnostic.
- Wendlandt, L., Kummerfeld, J. K., Mihalcea, R. (2018). *Factors Influencing the Surprising
  Instability of Word Embeddings.* NAACL-HLT. https://aclanthology.org/N18-1190/ — nearest-neighbour
  overlap is unstable even for words at 100–200 occurrences.
- Sahlgren, M., Lenci, A. (2016). *The Effects of Data Size and Frequency Range on Distributional
  Semantic Models.* EMNLP, 975–980. https://aclanthology.org/D16-1099/ — small data degrades DSMs
  unevenly across frequency bands.
- Lai, S., Liu, K., Xu, L., Zhao, J. (2016). *How to Generate a Good Word Embedding.* IEEE
  Intelligent Systems 31, 5–14. https://arxiv.org/abs/1507.05523 — "corpus domain is more important
  than corpus size" (about corpora used for *training*, hence analogy only for us).
