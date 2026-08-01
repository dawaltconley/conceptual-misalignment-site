import { z } from 'zod'
import type { WeightedNodeLinkData } from '~/types/networkx'
import {
  TermSchema as TermDataSchema,
  SourceSchema as SourceRefSchema,
} from './terms'

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

const SourceSchema = z.object({
  title: z.string(),
  description: z.string(),
  url: z.url(),
  cooccurrence: WeightedNodeLinkDataSchema.nullish(),
})

export const TermSchema = z.object({
  term: z.string(),
  sources: z.array(SourceSchema),
})

export type Term = z.infer<typeof TermSchema>

/**
 * A single per-(term, source) network file emitted by the pipeline's
 * `lib.NetworkData`: the term, the source it belongs to, and the graph itself
 * (null when the term never occurs in that source).
 */

export const NetworkDataSchema = z.object({
  term: TermDataSchema,
  source: SourceRefSchema,
  network: WeightedNodeLinkDataSchema.nullable(),
})

export type NetworkData = z.infer<typeof NetworkDataSchema>
