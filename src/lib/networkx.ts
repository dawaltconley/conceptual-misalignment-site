import { z } from 'zod'
import type { WeightedNodeLinkData } from '~/types/networkx'

const NodeId = z.union([z.string(), z.number()])

const GraphNode = z.looseObject({ id: NodeId })

const WeightedEdge = z.object({
  source: NodeId,
  target: NodeId,
  weight: z.number(),
})

export const WeightedNodeLinkDataSchema = z.object({
  directed: z.boolean(),
  multigraph: z.boolean(),
  graph: z.record(z.string(), z.unknown()),
  nodes: z.array(GraphNode),
  edges: z.array(WeightedEdge),
}) satisfies z.ZodType<WeightedNodeLinkData>
