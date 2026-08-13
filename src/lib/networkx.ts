import { z } from 'zod'
import type { WeightedNodeLinkData } from '~/types/networkx'
import { TermSchema, SourceSchema } from './terms'

const NodeId = z.union([z.string(), z.number()])

/**
 * `form` is the display glyph for the node's key (敎 -> 教); `variants` lists the
 * other words this node stands for — either the family the pipeline's
 * derivational merge folded in (`inspiration` carrying `inspire`), or, for a term
 * node, the lemmas its rendering matched here (`wisdom` carrying `wise`,
 * `wisely`). Both are optional labels — most nodes carry neither, though a term
 * node always carries `variants`, `[]` included.
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
