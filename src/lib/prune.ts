import type { NodeId, WeightedNodeLinkData } from '~/types/networkx'

/** `node id -> (neighbour id -> edge weight)`, undirected. */
type Adjacency = Map<NodeId, Map<NodeId, number>>

function buildAdjacency(data: WeightedNodeLinkData): Adjacency {
  const adjacency: Adjacency = new Map(data.nodes.map((n) => [n.id, new Map()]))
  for (const { source, target, weight } of data.edges) {
    adjacency.get(source)?.set(target, weight)
    adjacency.get(target)?.set(source, weight)
  }
  return adjacency
}

/**
 * Score a node by weighted proximity to `term`: its direct edge weight if there
 * is one, else the best two-hop weight product through a shared neighbour.
 */
function proximityScore(
  adjacency: Adjacency,
  term: NodeId,
  node: NodeId,
): number {
  const neighbours = adjacency.get(term)
  if (!neighbours) return 0
  const direct = neighbours.get(node)
  if (direct !== undefined) return direct

  let best = 0
  for (const [mid, weight] of neighbours) {
    const onward = adjacency.get(mid)?.get(node)
    if (onward !== undefined) best = Math.max(best, weight * onward)
  }
  return best
}

/**
 * Reduce a network to `centralNodeId` plus its `maxNodes` nearest neighbours,
 * dropping the weakest connections first.
 *
 * A port of the pipeline's `graph.prune.prune_to_neighborhood` (the build-side
 * `Pipeline.max_network_nodes`), so displaying a smaller graph gives the same
 * result as having exported one: direct neighbours are kept in order of edge
 * weight, any remaining capacity is filled with the highest-scoring two-hop
 * neighbours, and an edge survives only if both of its endpoints do.
 *
 * Returns `data` untouched when it is already small enough, when `maxNodes` is
 * not a positive number, or when the central node is absent from the graph.
 */
export default function pruneToNeighborhood<T extends WeightedNodeLinkData>(
  data: T,
  centralNodeId: NodeId,
  maxNodes?: number,
): T {
  if (!maxNodes || maxNodes < 1) return data
  if (data.nodes.length - 1 <= maxNodes) return data
  if (!data.nodes.some((n) => n.id === centralNodeId)) return data

  const adjacency = buildAdjacency(data)
  const oneHop = [...(adjacency.get(centralNodeId)?.entries() ?? [])]
    .sort(([, a], [, b]) => b - a)
    .map(([id]) => id)

  const keep = new Set<NodeId>([centralNodeId, ...oneHop.slice(0, maxNodes)])

  const remaining = maxNodes - (keep.size - 1)
  if (remaining > 0) {
    const twoHop = data.nodes
      .map((n) => n.id)
      .filter((id) => !keep.has(id))
      .map((id) => ({
        id,
        score: proximityScore(adjacency, centralNodeId, id),
      }))
      .filter(({ score }) => score > 0)
      .sort((a, b) => b.score - a.score)
    for (const { id } of twoHop.slice(0, remaining)) keep.add(id)
  }

  return {
    ...data,
    nodes: data.nodes.filter((n) => keep.has(n.id)),
    edges: data.edges.filter((e) => keep.has(e.source) && keep.has(e.target)),
  }
}
