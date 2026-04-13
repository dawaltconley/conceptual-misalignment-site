/**
 * Comprehensive TypeScript types for every possible output shape of
 * NetworkX's node_link_data() / json_graph.node_link_data().
 *
 * Full signature (NetworkX ≥ 3.x):
 *
 *   nx.node_link_data(
 *     G,
 *     attrs   = None,
 *     source  = "source",   // key used for source node id in each edge dict
 *     target  = "target",   // key used for target node id in each edge dict
 *     name    = "id",       // key used for node id in each node dict
 *     key     = "key",      // key used for edge key in multigraph edge dicts
 *     edges   = "edges",    // top-level key for the edge array ("links" in older releases)
 *     nodes   = "nodes",    // top-level key for the node array
 *   )
 *
 * Every type parameter below corresponds to one of these kwargs so that
 * callers can narrow the type to exactly what their invocation produces.
 */

// ---------------------------------------------------------------------------
// Primitive
// ---------------------------------------------------------------------------

/**
 * NetworkX node IDs can be any Python hashable. After JSON round-trip they
 * arrive as strings or numbers (booleans and None are technically possible
 * but extremely rare in practice).
 */
export type NodeId = string | number;

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

/**
 * A node object in the "nodes" array.
 *
 * @template TNameKey  The key that holds the node id (maps to the `name` kwarg, default "id").
 * @template TAttrs    Any additional per-node attributes stored on the graph.
 *
 * @example
 *   // Default: { id: NodeId }
 *   type N = GraphNode;
 *
 *   // Custom name key and extra attrs:
 *   type N = GraphNode<"label", { color: string; weight: number }>;
 */
export type GraphNode<
  TNameKey extends string = "id",
  TAttrs extends Record<string, unknown> = Record<string, unknown>,
> = { [K in TNameKey]: NodeId } & TAttrs;

// ---------------------------------------------------------------------------
// Edges
// ---------------------------------------------------------------------------

/**
 * An edge object for a simple (non-multigraph) graph.
 *
 * @template TSrcKey   Key for the source node id (maps to `source` kwarg, default "source").
 * @template TTgtKey   Key for the target node id (maps to `target` kwarg, default "target").
 * @template TAttrs    Any additional per-edge attributes (e.g. weight, capacity).
 *
 * @example
 *   // Default: { source: NodeId; target: NodeId }
 *   type E = SimpleEdge;
 *
 *   // With a weight attribute:
 *   type E = SimpleEdge<"source", "target", { weight: number }>;
 */
export type SimpleEdge<
  TSrcKey extends string = "source",
  TTgtKey extends string = "target",
  TAttrs extends Record<string, unknown> = Record<string, unknown>,
> = { [K in TSrcKey]: NodeId } & { [K in TTgtKey]: NodeId } & TAttrs;

/**
 * An edge object for a multigraph. Identical to SimpleEdge but adds the edge
 * key field that distinguishes parallel edges.
 *
 * @template TSrcKey   Key for the source node id (default "source").
 * @template TTgtKey   Key for the target node id (default "target").
 * @template TKeyKey   Key for the edge key   (maps to `key` kwarg, default "key").
 * @template TAttrs    Any additional per-edge attributes.
 */
export type MultiEdge<
  TSrcKey extends string = "source",
  TTgtKey extends string = "target",
  TKeyKey extends string = "key",
  TAttrs extends Record<string, unknown> = Record<string, unknown>,
> = SimpleEdge<TSrcKey, TTgtKey, TAttrs> & { [K in TKeyKey]: NodeId };

// ---------------------------------------------------------------------------
// Top-level graph object
// ---------------------------------------------------------------------------

/**
 * The object returned by nx.node_link_data() in its most general form.
 *
 * All six customisable kwargs are modelled as type parameters so the type
 * can be narrowed to exactly the shape your call produces.
 *
 * @template TEdgeKey  Top-level key for the edge array  (maps to `edges` kwarg).
 *                     Default is "edges" (NetworkX ≥ 3.4).
 *                     Use "links" for older NetworkX or when passed explicitly.
 * @template TNodeKey  Top-level key for the node array  (maps to `nodes` kwarg, default "nodes").
 * @template TEdge     Shape of each edge object — use SimpleEdge or MultiEdge (with their own
 *                     type params) to customise source/target/key field names and attrs.
 * @template TNode     Shape of each node object — use GraphNode (with its own type params)
 *                     to customise the id field name and attrs.
 *
 * @example
 *   // Fully defaulted — matches the most common output:
 *   const g: NodeLinkData = JSON.parse(fs.readFileSync("graph.json", "utf8"));
 *
 *   // Older NetworkX that emits "links" instead of "edges":
 *   type LegacyGraph = NodeLinkData<"links">;
 *
 *   // Multigraph with weighted edges and labelled nodes:
 *   type RichGraph = NodeLinkData<
 *     "edges",
 *     "nodes",
 *     MultiEdge<"source", "target", "key", { weight: number }>,
 *     GraphNode<"id", { label: string; community: number }>
 *   >;
 */
export type NodeLinkData<
  TEdgeKey extends string = "edges",
  TNodeKey extends string = "nodes",
  TEdge extends SimpleEdge = SimpleEdge,
  TNode extends GraphNode = GraphNode,
> = {
  /** Whether the graph is directed. */
  directed: boolean;
  /** Whether the graph is a multigraph (allows parallel edges). */
  multigraph: boolean;
  /**
   * Graph-level attribute dict. May contain any JSON-serialisable values;
   * empty ({}) when no graph attributes have been set.
   */
  graph: Record<string, unknown>;
} & { [K in TNodeKey]: TNode[] } &
  { [K in TEdgeKey]: TEdge[] };

// ---------------------------------------------------------------------------
// Convenience aliases for the two most common cases
// ---------------------------------------------------------------------------

/**
 * Simple undirected or directed graph with default key names and weighted edges.
 * Matches the output of nx.node_link_data(G) on NetworkX ≥ 3.4.
 */
export type WeightedNodeLinkData = NodeLinkData<
  "edges",
  "nodes",
  SimpleEdge<"source", "target", { weight: number }>
>;

/**
 * Same as WeightedNodeLinkData but with "links" as the edge array key.
 * Matches older NetworkX defaults and explicit edges="links" calls.
 */
export type WeightedNodeLinkDataLegacy = NodeLinkData<
  "links",
  "nodes",
  SimpleEdge<"source", "target", { weight: number }>
>;
