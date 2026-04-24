import type { NodeId, WeightedNodeLinkData, SimpleEdge } from '~/types/networkx'
import type { Dictionary } from '@build/cedict'
import { useState, useEffect, useRef } from 'react'
import useSize from '@lib/browser/hooks/useSize'
import clsx from 'clsx'
import * as d3 from 'd3'
import { isNotEmpty } from '@lib/utils'
import HanziNode from '@components/HanziNode'

const COLLISION_RADIUS = 6

export interface NetworkProps {
  data: WeightedNodeLinkData
  centralNodeId: NodeId
  dictionary?: Dictionary
}

export default function Network({
  data,
  centralNodeId,
  dictionary,
}: NetworkProps): JSX.Element {
  const [nodes, setNodes] = useState<Node[]>([])
  const containerRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const simulationRef = useRef<d3.Simulation<Node, Link> | null>(null)
  const draggingId = useRef<NodeId | null>(null)

  const size = useSize(containerRef)
  const { width = 0, height = 0 } = size || {}

  // Run D3 simulation in normalized 0–100 coordinate space
  useEffect(() => {
    const nodes = data.nodes.map<Node>((n) =>
      n.id === centralNodeId ? { ...n, fx: 50, fy: 50 } : { ...n },
    )
    const links = data.edges.map<Link>((e) => ({ ...e, value: e.weight }))

    const simulation = d3
      .forceSimulation(nodes)
      .force(
        'link',
        d3
          .forceLink<Node, Link>(links)
          .id((d) => d.id)
          .distance(8),
      )
      .force('collide', d3.forceCollide().radius(COLLISION_RADIUS))
      .force('charge', d3.forceManyBody().strength(-1))
      .force('center', d3.forceCenter(50, 50))
      .alphaDecay(0.05)
      .velocityDecay(0.5)
      .on('tick', () => setNodes([...nodes]))

    simulationRef.current = simulation
    return () => {
      simulation.stop()
    }
  }, [data, centralNodeId])

  // Redraw edges on canvas whenever node positions or container size change
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    drawEdges(getEdges(nodes, data.edges), canvas, size)
  }, [nodes, data, size])

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
    if (!draggingId.current || !containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
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
    <div
      ref={containerRef}
      className="relative aspect-square min-h-96 w-full overflow-hidden"
    >
      <canvas
        ref={canvasRef}
        className="absolute inset-0"
        style={{ width, height }}
      />
      {nodes.map((node) => (
        <div
          key={node.id}
          className={clsx(
            'absolute rounded-full px-1.5 py-0.5 text-xs shadow-sm ring-1 ring-gray-200',
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
          {dictionary?.[String(node.id)]
            ? <HanziNode id={node.id} entry={dictionary[String(node.id)]} isCentral={node.id === centralNodeId} />
            : String(node.id)}
        </div>
      ))}
    </div>
  )
}

interface Node extends d3.SimulationNodeDatum {
  id: NodeId
}

interface Link extends d3.SimulationLinkDatum<Node> {
  source: NodeId
  target: NodeId
  value: number
}

interface Edge {
  x1: number
  y1: number
  x2: number
  y2: number
  weight?: number
}

function getEdges(
  nodes: Node[],
  edges: SimpleEdge<'source', 'target', { weight: number }>[],
): Edge[] {
  const nodeMap = new Map(nodes.map((n) => [n.id, n]))
  return edges
    .map<Edge | null>(({ source, target, weight }) => {
      const { x: x1, y: y1 } = nodeMap.get(source) || {}
      const { x: x2, y: y2 } = nodeMap.get(target) || {}
      if (!x1 || !x2 || !y1 || !y2) {
        return null
      }
      return { x1, y1, x2, y2, weight }
    })
    .filter(isNotEmpty)
}

function drawEdges(
  edges: Edge[],
  canvas: HTMLCanvasElement,
  size?: { width: number; height: number },
): void {
  const { width = canvas.width, height = canvas.height } = size || {}
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

  for (const { x1, y1, x2, y2, weight = 1 } of edges) {
    ctx.beginPath()
    ctx.strokeStyle = '#9ca3af'
    ctx.lineWidth = Math.log2(weight + 1) * 0.4
    ctx.moveTo(toX(x1), toY(y1))
    ctx.lineTo(toX(x2), toY(y2))
    ctx.stroke()
  }
}
