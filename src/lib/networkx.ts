import { z } from 'zod'
import type { WeightedNodeLinkData } from '~/types/networkx'
import { TermSchema, SourceSchema } from './terms'

const NodeId = z.union([z.string(), z.number()])

/**
 * `form` is the display glyph for the node's key (敎 -> 教); `variants` lists the
 * words the pipeline's derivational merge folded into this node (`inspiration`
 * carrying `inspire`). Both are optional labels — most nodes carry neither.
 */
const GraphNode = z.looseObject({
  id: NodeId,
  form: z.string().optional(),
  variants: z.array(z.string()).optional(),
})

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

/**
 * A single per-(term, source) network file emitted by the pipeline's
 * `lib.NetworkData`: the term, the source it belongs to, and the graph itself
 * (null when the term never occurs in that source).
 */

export const NetworkDataSchema = z.object({
  term: TermSchema,
  source: SourceSchema,
  network: WeightedNodeLinkDataSchema.nullable(),
})

export type NetworkData = z.infer<typeof NetworkDataSchema>
