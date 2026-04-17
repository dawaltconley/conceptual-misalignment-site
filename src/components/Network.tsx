import type { NodeId, WeightedNodeLinkData, SimpleEdge } from '~/types/networkx'
import {
  useState,
  useEffect,
  useLayoutEffect,
  useRef,
  type RefObject,
} from 'react'
import useResizeObserver from '@react-hook/resize-observer'
import twColors from 'tailwindcss/colors'
import clsx from 'clsx'
import * as d3 from 'd3'
import { isNotEmpty } from '@lib/utils'

interface NetworkProps {
  data: WeightedNodeLinkData
  centralNodeId: NodeId
}

export default function Network({
  data,
  centralNodeId,
}: NetworkProps): JSX.Element {
  const [nodes, setNodes] = useState<Node[]>([])
  const containerRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  const size = useSize(containerRef)
  const { width = 0, height = 0 } = size || {}
  // Run D3 simulation in normalized 0–100 coordinate space
  useEffect(() => {
    const nodes = data.nodes.map<Node>((n) =>
      n.id === centralNodeId ? { ...n, fx: 50, fy: 50 } : { ...n },
    )
    const links = data.edges.map<Link>((e) => ({ ...e, value: e.weight }))

    d3.forceSimulation(nodes)
      .force(
        'link',
        d3
          .forceLink<Node, Link>(links)
          .id((d) => d.id)
          .distance(8),
      )
      .force('collide', d3.forceCollide().radius(COLLISION_RADIUS))
      .force('charge', d3.forceManyBody().strength(-4))
      .force('center', d3.forceCenter(50, 50))
      .alphaDecay(0.05)
      .velocityDecay(0.5)
      .on('tick', () => setNodes([...nodes]))
  }, [data, centralNodeId])

  // Redraw edges on canvas whenever node positions or container size change
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    drawEdges(getEdges(nodes, data.edges), canvas, size)
  }, [nodes, data, size])

  return (
    <div ref={containerRef} className="relative aspect-square min-h-96 w-full">
      <canvas
        ref={canvasRef}
        className="absolute inset-0"
        style={{ width, height }}
      />
      {nodes.map((node) => (
        <div
          key={node.id}
          className={clsx(
            'absolute rounded-full bg-white px-1.5 py-0.5 text-xs shadow-sm ring-1 ring-gray-200',
            node.id === centralNodeId &&
              'bg-red-500 text-white ring-transparent',
          )}
          style={{
            transform: `translate(calc(${(node.x ?? 0) * 0.01 * width}px - 50%), calc(${(node.y ?? 0) * 0.01 * height}px - 50%))`,
          }}
        >
          {node.id}
        </div>
      ))}
    </div>
  )
}

function useSize(target: RefObject<Element>): DOMRect | undefined {
  const [size, setSize] = useState<DOMRect>()
  useLayoutEffect(() => {
    if (target.current) {
      setSize(target.current.getBoundingClientRect())
    }
  }, [target])
  useResizeObserver(target, (entry) => setSize(entry.contentRect))
  return size
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
    ctx.strokeStyle = twColors.gray['400']
    ctx.lineWidth = Math.log2(weight + 1) * 0.4
    ctx.moveTo(toX(x1), toY(y1))
    ctx.lineTo(toX(x2), toY(y2))
    ctx.stroke()
  }
}

interface Node extends d3.SimulationNodeDatum {
  id: NodeId
}

interface Link extends d3.SimulationLinkDatum<Node> {
  source: NodeId
  target: NodeId
  value: number
}

const COLLISION_RADIUS = 6
