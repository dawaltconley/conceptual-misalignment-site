import type { EmbeddingNode } from './embeddings'

/**
 * How to order the words *within* a Louvain community, for the scatter legend
 * and the community dialog. See `notes/community-legend-ordering.md`, which
 * measures all of these against the shipped SEP run.
 *
 * - `proximity` — cosine to the community's **virtue anchor**: the mean of its
 *   target vectors, or its centroid where no target lands in it. The default.
 *   Measured best: in the 10 SEP communities that hold a target it surfaces the
 *   concept (`trust, distrust, reliability, betray` for trustworthiness) where
 *   the alternatives surface the register band (`trustor, wrongdoer,
 *   interrogator, employer` — the agentive noun slot). In the 11 that hold no
 *   target the anchor *is* the centroid, so it degrades into `prototypicality`
 *   exactly and only there.
 * - `prototypicality` — cosine to the community centroid: the most
 *   representative member, whatever the community turns out to be about.
 * - `strength` — weighted degree in the similarity graph (the previous default).
 *   Nearly interchangeable with pagerank (ρ +0.99), so pagerank is not offered.
 * - `alphabetical` — the original behaviour, kept for comparison.
 *
 * `silhouette` is deliberately absent: it correlates with `prototypicality` at
 * ρ +0.90 within communities, so offering both would show the same list twice.
 */
export type CommunitySort =
  | 'proximity'
  | 'prototypicality'
  | 'strength'
  | 'alphabetical'

/** One-line description of an ordering, for the community dialog. */
export const COMMUNITY_SORT_LABELS: Record<CommunitySort, string> = {
  proximity:
    'closeness to the community’s virtue (or its centre, if it has no virtue)',
  prototypicality: 'closeness to the community’s centre',
  strength: 'embedding strength',
  alphabetical: 'name',
}

function l2Normalize(vec: number[]): number[] {
  let sum = 0
  for (const v of vec) sum += v * v
  const norm = Math.sqrt(sum)
  // The exported vectors are PCA-reduced and *not* unit-norm, so normalizing is
  // what makes the dot products below true cosines rather than a length ranking.
  return norm > 1e-12 ? vec.map((v) => v / norm) : vec.map(() => 0)
}

function dot(a: number[], b: number[]): number {
  let sum = 0
  for (let i = 0; i < Math.min(a.length, b.length); i++) sum += a[i] * b[i]
  return sum
}

/** Unit-normalized mean of already-unit vectors (an empty set gives `null`). */
function meanDirection(vectors: number[][]): number[] | null {
  if (!vectors.length) return null
  const dims = vectors[0].length
  const sum = new Array<number>(dims).fill(0)
  for (const v of vectors) for (let i = 0; i < dims; i++) sum[i] += v[i]
  return l2Normalize(sum.map((s) => s / vectors.length))
}

/**
 * Group `nodes` by community and order each group by `sort`, descending.
 *
 * The result is indexed by community id, matching how the legend and dialogs
 * look groups up. Isolates (community `-1`) land on a non-index property and so
 * stay out of the legend, exactly as before.
 */
export function groupCommunities(
  nodes: EmbeddingNode[],
  sort: CommunitySort = 'proximity',
): EmbeddingNode[][] {
  const groups: EmbeddingNode[][] = []
  for (const n of nodes) {
    groups[n.community] ??= []
    groups[n.community].push(n)
  }

  if (sort === 'alphabetical') {
    for (const group of groups) group?.sort((a, b) => a.id.localeCompare(b.id))
    return groups
  }
  if (sort === 'strength') {
    for (const group of groups) group?.sort((a, b) => b.strength - a.strength)
    return groups
  }

  const units = new Map(nodes.map((n) => [n.id, l2Normalize(n.vec)]))
  for (const group of groups) {
    if (!group) continue
    const vectors = group.map((n) => units.get(n.id)!)
    const centroid = meanDirection(vectors)
    // The anchor is what separates the two vector orderings: proximity leans on
    // the community's targets when it has any, and is identical to
    // prototypicality when it does not.
    const anchor =
      sort === 'proximity'
        ? (meanDirection(
            group.filter((n) => n.target).map((n) => units.get(n.id)!),
          ) ?? centroid)
        : centroid
    if (!anchor) continue
    const score = new Map(
      group.map((n) => [n.id, dot(units.get(n.id)!, anchor)]),
    )
    group.sort((a, b) => (score.get(b.id) ?? 0) - (score.get(a.id) ?? 0))
  }
  return groups
}
