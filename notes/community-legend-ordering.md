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

Compute similarities on the full 50-d `vec` (not the 2-D projection); L2-normalize first so
it's true cosine (the reduced vectors aren't unit-norm).

Relevant code: `src/components/EmbeddingScatter.tsx` (`communityLegend`),
`src/components/ScatterLegend.tsx`; pipeline export in `scripts/embed.py`
(`lib.Embeddings.from_matrix`, `src/lib/embeddings.ts` schema) if adding `freq`.
