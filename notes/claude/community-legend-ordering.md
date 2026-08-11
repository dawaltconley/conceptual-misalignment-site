# Ordering words within a Louvain community (scatter legend)

The embedding-scatter legend currently lists each community's words **alphabetically**
(`terms.sort()` in `EmbeddingScatter.tsx` → `communityLegend`). Alphabetical is
uninformative; we want a metric that surfaces the "important" or the "distinctive"
members first. Options below, split by what they answer and by cost.

## What each node already carries

`public/embeddings/*.json` nodes are `{ id, target, community, vec }`, where `vec` is the
PCA-reduced (50-d, variance-ordered) pooled vector. So **anything computable from `vec` +
`community` is free** (client-side, in the legend `useMemo`). Frequency or graph-structure
metrics need a new field exported from the pipeline (`scripts/embed.py` /
`lib.Embeddings.from_matrix`).

## A. "Most significant / central" (importance)

Free from `vec`:
- **Prototypicality** — cosine similarity of the word's `vec` to its **community centroid**
  (mean of that community's vectors). Descending → most *representative* word first.
  Simple, on-theme, consistent with what's plotted.
- **Proximity to the virtue** — cosine similarity to the nearest **target** (仁/義 or the
  English rendering). Ranks by closeness to the virtue itself (= the research question), but
  only meaningful where a target anchors the community; needs a centroid fallback elsewhere.

Needs a new exported field:
- **Corpus frequency** — how often the word occurs. Most natural "most represented"; we
  already count it (`build_vocab` frequency / `by_word` lengths), so it's a one-field add to
  the embeddings export. Caveat: counts are heavy-tailed and differ in scale between Chinese
  and English — rank or log-frequency reads better.
- **Weighted degree (strength)** in the cosine kNN graph — sum of a node's edge weights.
  High = dense/tight region ("hub"). Since kNN gives everyone ~k edges, the *weights*
  discriminate. Per-node export from the graph.
- **PageRank / eigenvector centrality** on that graph — global importance. Per-node export.

## B. "Most unique / specific to the community" (distinctiveness)

Free from `vec`:
- **Silhouette-style specificity** — `s = sim(word, own centroid) − max_over_other_communities
  sim(word, that centroid)`. Descending → the word that is distinctively this community and
  not the others. Most direct answer to "unique to that community," and free.

Needs graph / co-occurrence data:
- **Community core-ness (participation)** — fraction of a node's edge weight that stays
  *inside* its community vs. bridges out. High = core member; low = connector. Needs the graph.
- **TF-IDF-style specificity** — words co-occurring with this community's theme but not
  others. Most faithful to "specific," but needs co-occurrence counts, not just embeddings.

## Recommendation

The two that are **free, on-theme, and answer each framing directly**:
- **Prototypicality** (importance) and **silhouette specificity** (uniqueness) — both
  computable in the legend from data already shipped, no pipeline change. Wire as a small
  sort toggle (alphabetical / representative / distinctive) to compare.
- For a corpus-grounded "most represented," add a **`freq`** field to the embeddings export
  (cheapest high-value pipeline change; rank or log it).

## Measured against the post-debiasing SEP run (2026-08-10)

Every metric above, computed over the shipped artifact (3455 nodes, 21 communities,
17 targets) with `scripts/tools/community_ordering.py --corpus sep`. Within-community
Spearman, so it measures the ranking the legend actually shows.

**The recommendation above pairs two metrics that turn out to be the same metric.**
Prototypicality and silhouette correlate at **+0.90** — as a toggle they would give the
reader the same list twice. Strength and pagerank are worse, at **+0.99**: pick either,
never both. Eigenvector is only weakly tied to anything (+0.13…+0.44) but earns its
independence by surfacing oddities rather than exemplars — `patrilineage`, `psychopath`,
`piano`, `trolley`, `noun`, `grammarian` are its per-community winners. Not a legend.

**`proximity_to_virtue` is the one to default to.** Two independent reasons:

1. It **degrades into prototypicality exactly where it has to** and nowhere else.
   Verified: the two rankings are identical in 11 of 21 communities, and those 11 are
   precisely the target-free ones (the anchor falls back to the centroid by
   construction). So choosing it costs nothing in the communities it cannot speak to.
2. In the 10 communities that **do** hold a target it is dramatically better, because it
   inherits the local geometry rather than the noisy partition — the same asymmetry
   recorded in [[sep-community-register-domination]], where per-target neighbourhoods
   are coherent while the community around them often is not:

| community | prototypicality says | proximity_to_virtue says |
|---|---|---|
| C5 *trustworthiness* | trustor, wrongdoer, interrogator, employer, perpetrator | **trust, distrust, reliability, unreliable, betray** |
| C14 *faith ritual righteousness* | sermon, textual, scripture, poem, prose | **prayer, religion, salvation, divine, rite** |
| C6 *knowledge* | conceptualization, imagine, conception, vision | **knowing, ignorance, awareness, learning, foreknowledge** |
| C12 *wisdom* | apartheid, immigrant, indigenous, woman, racism | **courage, virtue, philosophizing, advice** |

Prototypicality is reporting the register band; proximity is reporting the concept.

**The genuinely orthogonal second axis is frequency** (ρ −0.03…+0.08 against everything,
including proximity at −0.02) — so the useful toggle is *proximity / frequency*, not
*prototypicality / silhouette*.

### Why frequency did not register, and the `freq` export (done)

Two separate faults:

1. **There was no corpus-frequency field in the export at all.** `Embeddings.from_matrix`
   wrote `doc_freq`, `norm`, `strength`, `pagerank`, `eigenvector` — never the raw count.
   `community_ordering.py` read frequency from a **conllu**, which only Mengzi has, so
   every SEP word scored 0.0: a dead column, and a row of +0.00 across the correlation
   matrix that made frequency look uncorrelated when it was simply absent.
2. **`doc_freq` is a saturating stand-in**, bounded by the corpus's document count. On the
   184-document SEP run, **85% of nodes sat in a tie group of 10 or more** and the maximum
   was 184 — every document. Ties then fall back to insertion order.

**Fixed:** `Vector.freq` is now exported (`src/lib/embeddings.ts` matches, defaulted so
pre-existing artifacts still validate). It costs nothing to compute — `embed()` already
had the counts (`content_frequencies` → `freq`, handed to `families.merge_map(counts=…)`)
and was discarding them. Under a variant merge the counts are **summed**, which is exact
for occurrences though not for document frequencies, so no second corpus pass is needed
(see `embeddings.occurrences.content_frequencies`).

Measured on a 17-article verification run: `freq` gives 178 distinct values against
`doc_freq`'s 17, and 51% of nodes in a 10+ tie group against doc_freq's 100%. Merges fold
in correctly (`action` = 434 absorbing `act`; `ability` = 157 absorbing `able`).
`community_ordering.py` now prefers the exported `freq`, falls back to the conllu, and
only then to `doc_freq` — printing which source it used, since an artifact predating this
change silently lands on the saturating one.

The counts are heavy-tailed (`morality` 882 vs a floor of 3), so **log or rank before
sorting a legend by them**.

Compute similarities on the full 50-d `vec` (not the 2-D projection); L2-normalize first so
it's true cosine (the reduced vectors aren't unit-norm).

Relevant code: `src/components/EmbeddingScatter.tsx` (`communityLegend`),
`src/components/ScatterLegend.tsx`; pipeline export in `scripts/embed.py`
(`lib.Embeddings.from_matrix`, `src/lib/embeddings.ts` schema) if adding `freq`.
