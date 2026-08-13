import type { NodeId, WeightedNodeLinkData, SimpleEdge } from '~/types/networkx'
import type { Dictionary } from '@build/cedict'
import type { Size, Range, Line } from '@lib/graphs'
import {
  useState,
  useEffect,
  useMemo,
  useRef,
  forwardRef,
  type ReactNode,
} from 'react'
import { useResizeObserver } from 'use-resize-observer'
import clsx from 'clsx'
import * as d3 from 'd3'
import { isNotEmpty } from '@lib/utils'
import { isPoint, getNormalizer } from '@lib/graphs'
import pruneToNeighborhood from '@lib/prune'
import HanziNode from '@components/HanziNode'
import EnglishNode from '@components/EnglishNode'

const COLLISION_RADIUS = 4

export interface NetworkProps {
  data: WeightedNodeLinkData
  centralNodeId: NodeId
  /**
   * Cap on how many nodes *besides* the central one are drawn, thinning the
   * graph from the weakest connections up (see `@lib/prune`). Unset draws the
   * network as exported.
   */
  maxNodes?: number
  actualEdgeWeightRange?: Range
  targetEdgeWeightRange?: Range
  dictionary?: Dictionary
}

export default function Network({
  data,
  centralNodeId,
  maxNodes,
  actualEdgeWeightRange = { min: 1, max: 3 },
  targetEdgeWeightRange = { min: 1, max: 5 },
  dictionary = {},
}: NetworkProps): JSX.Element {
  const [nodes, setNodes] = useState<Node[]>([])
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const simulationRef = useRef<d3.Simulation<Node, Link> | null>(null)
  const draggingId = useRef<NodeId | null>(null)

  const {
    ref: containerRef,
    width = 0,
    height = 0,
  } = useResizeObserver<HTMLDivElement>()

  const graph = useMemo(
    () => pruneToNeighborhood(data, centralNodeId, maxNodes),
    [data, centralNodeId, maxNodes],
  )

  // Run D3 simulation in normalized 0–100 coordinate space
  useEffect(() => {
    const nodes = graph.nodes.map<Node>((n) =>
      n.id === centralNodeId ? { ...n, fx: 50, fy: 50 } : { ...n },
    )
    const links = graph.edges.map<Link>((e) => ({ ...e, value: e.weight }))
    const values = links.map((l) => l.value)
    const min = Math.min(...values)
    const max = Math.max(...values)

    const toLinkDistance = getNormalizer({ min, max }, { min: 40, max: 5 })
    const toLinkStrength = getNormalizer({ min, max }, { min: 0.7, max: 1 })

    const simulation = d3
      .forceSimulation(nodes)
      .force(
        'link',
        d3
          .forceLink<Node, Link>(links)
          .id((d) => d.id)
          .distance((d) => toLinkDistance(d.value))
          .strength((d) => toLinkStrength(d.value)),
      )
      .force(
        'collide',
        d3.forceCollide().radius(COLLISION_RADIUS).strength(0.2),
      )
      .force('charge', d3.forceManyBody().strength(-0.5))
      .force('center', d3.forceCenter(50, 50).strength(0.1))
      .alphaDecay(0.05)
      .velocityDecay(0.5)
      .on('tick', () => setNodes([...nodes]))

    simulationRef.current = simulation
    return () => {
      simulation.stop()
    }
  }, [graph, centralNodeId])

  // Redraw edges on canvas whenever node positions or container size change
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const edges = getEdges(nodes, graph.edges, {
      toWidth: getNormalizer(actualEdgeWeightRange, targetEdgeWeightRange),
      toOpacity: getNormalizer(actualEdgeWeightRange, { min: 0, max: 1 }),
    })
    drawEdges(edges, canvas, { width, height })
  }, [nodes, graph, width, height])

  function handlePointerDown(
    e: React.PointerEvent<HTMLDivElement>,
    nodeId: NodeId,
  ) {
    if (nodeId === centralNodeId) return
    e.currentTarget.setPointerCapture(e.pointerId)
    draggingId.current = nodeId
    const node = nodes.find((n) => n.id === nodeId)
    if (node) {
      node.fx = node.x
      node.fy = node.y
    }
    simulationRef.current?.alphaTarget(0.3).restart()
  }

  function handlePointerMove(e: React.PointerEvent<HTMLDivElement>) {
    if (!draggingId.current || !canvasRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width) * 100
    const y = ((e.clientY - rect.top) / rect.height) * 100
    const node = nodes.find((n) => n.id === draggingId.current)
    if (node) {
      node.fx = x
      node.fy = y
    }
  }

  function handlePointerUp(
    _e: React.PointerEvent<HTMLDivElement>,
    nodeId: NodeId,
  ) {
    if (draggingId.current !== nodeId) return
    const node = nodes.find((n) => n.id === nodeId)
    if (node) {
      node.fx = undefined
      node.fy = undefined
    }
    draggingId.current = null
    simulationRef.current?.alphaTarget(0)
  }

  return (
    <Wrapper ref={containerRef}>
      <canvas
        ref={canvasRef}
        className="absolute inset-0"
        style={{ width, height }}
      />
      {nodes.map((node) => (
        <div
          key={node.id}
          className={clsx(
            'absolute rounded-full px-2.5 py-0.5 text-lg shadow-sm ring-1 ring-gray-200',
            node.id === centralNodeId
              ? 'z-10 cursor-default bg-red-500 text-white ring-transparent'
              : 'cursor-grab select-none bg-white active:cursor-grabbing',
          )}
          style={{
            transform: `translate(calc(${(node.x ?? 0) * 0.01 * width}px - 50%), calc(${(node.y ?? 0) * 0.01 * height}px - 50%))`,
          }}
          onPointerDown={(e) => handlePointerDown(e, node.id)}
          onPointerMove={handlePointerMove}
          onPointerUp={(e) => handlePointerUp(e, node.id)}
        >
          {dictionary[node.id.toString()] ? (
            <HanziNode
              id={node.id}
              entry={dictionary[node.id.toString()]}
              isCentral={node.id === centralNodeId}
            />
          ) : (
            <EnglishNode
              id={node.id}
              variants={node.variants}
              isCentral={node.id === centralNodeId}
            />
          )}
        </div>
      ))}
    </Wrapper>
  )
}

export const NetworkSkeleton = (): JSX.Element => (
  <Wrapper>
    <div className="absolute inset-0 m-auto flex items-center justify-center text-gray-500">
      Loading…
    </div>
    <div className="skeleton absolute inset-0" />
  </Wrapper>
)

export const Wrapper = forwardRef<HTMLDivElement, { children: ReactNode }>(
  ({ children }, ref) => (
    <div
      ref={ref}
      className="relative aspect-square min-h-96 w-full touch-manipulation overflow-hidden"
    >
      {children}
    </div>
  ),
)

interface Node extends d3.SimulationNodeDatum {
  id: NodeId
  /** Words the pipeline's derivational merge folded into this node, if any. */
  variants?: string[]
}

interface Link extends d3.SimulationLinkDatum<Node> {
  source: NodeId
  target: NodeId
  value: number
}

interface Edge extends Line {
  lineWidth: number
  opacity: number
}

interface GetEdgesOpts {
  toWidth?: (n: number) => number
  toOpacity?: (n: number) => number
}

function getEdges(
  nodes: Node[],
  edges: SimpleEdge<'source', 'target', { weight: number }>[],
  { toWidth = (n) => n, toOpacity = (n) => n }: GetEdgesOpts = {},
): Edge[] {
  const nodeMap = new Map(nodes.map((n) => [n.id, n]))
  return edges
    .map<Edge | null>(({ source, target, weight }) => {
      const start = nodeMap.get(source) || {}
      const end = nodeMap.get(target) || {}
      if (!isPoint(start) || !isPoint(end)) {
        return null
      }
      return {
        start,
        end,
        lineWidth: toWidth(weight),
        opacity: toOpacity(weight),
      }
    })
    .filter(isNotEmpty)
}

function drawEdges(
  edges: Edge[],
  canvas: HTMLCanvasElement,
  canvasSize?: Size,
): void {
  const { width = canvas.width, height = canvas.height } = canvasSize || {}
  const dpr = window.devicePixelRatio || 1
  canvas.width = width * dpr
  canvas.height = height * dpr

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.scale(dpr, dpr)
  ctx.clearRect(0, 0, width, height)

  // Convert normalized 0–100 coords to CSS pixels
  const toX = (v: number) => (v / 100) * width
  const toY = (v: number) => (v / 100) * height

  for (const { start, end, lineWidth, opacity } of edges) {
    ctx.beginPath()
    ctx.strokeStyle = '#9ca3af'
    ctx.globalAlpha = Math.min(1, Math.max(0, opacity))
    ctx.lineWidth = lineWidth
    ctx.moveTo(toX(start.x), toY(start.y))
    ctx.lineTo(toX(end.x), toY(end.y))
    ctx.stroke()
  }
}
