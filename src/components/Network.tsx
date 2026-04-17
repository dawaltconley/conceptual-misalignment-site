import type { NodeId, WeightedNodeLinkData } from '~/types/networkx'
import { useState, useEffect, useRef } from 'react'
import * as d3 from 'd3'

interface NetworkProps {
  centralNodeId?: NodeId
}

export default function Network({
  centralNodeId = 'benevolence',
}: NetworkProps): JSX.Element {
  const [data, setData] = useState<Node[]>()
  const svg = useRef<SVGSVGElement | null>(null)

  // consider useLayoutEffect?
  useEffect(() => {
    const nodes = DATA.nodes.map<Node>((n) => {
      if (n.id === centralNodeId) {
        // fix the central node in the center
        return { ...n, fx: 50, fy: 50 }
      }
      return { ...n }
    })
    const links = DATA.edges.map<Link>((e) => ({ ...e, value: e.weight }))
    d3.forceSimulation(nodes)
      .force(
        'link',
        d3
          .forceLink<Node, Link>(links)
          .id((d) => d.id)
          .distance(8),
      )
      .force('collide', d3.forceCollide().radius(RADIUS * 2))
      .force('charge', d3.forceManyBody().strength(-4))
      .force('center', d3.forceCenter(50, 50))
      .alphaDecay(0.05)
      .velocityDecay(0.5)
      .on('tick', () => {
        setData([...nodes])
      })
  }, [centralNodeId])

  return (
    <div className="min-h-96 w-full">
      <svg ref={svg} viewBox="0 0 100 100" className="h-full w-full">
        <g stroke="gray" fill="white" strokeWidth={0.2}>
          {DATA?.edges.map(({ source, target, weight }) => {
            if (!data) return null
            const start = data.find((n) => n.id === source)
            const end = data.find((n) => n.id === target)
            if (!start || !end) return null
            return (
              <line
                x1={start.x}
                y1={start.y}
                x2={end.x}
                y2={end.y}
                strokeWidth={Math.log2(weight + 1) * 0.2}
              />
            )
          })}
          {data?.map((node) => (
            <>
              <circle cx={node.x} cy={node.y} r={RADIUS} />
              <text
                x={node.x}
                y={node.y}
                fontSize={1}
                strokeWidth={0.02}
                transform="translate(-50%, -50%)"
                textAnchor="middle"
                stroke={node.id === centralNodeId ? 'red' : 'black'}
                fill={node.id === centralNodeId ? 'red' : 'black'}
              >
                {node.id}
              </text>
            </>
          ))}
        </g>
      </svg>
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

const RADIUS = 3

const DATA: WeightedNodeLinkData = {
  directed: false,
  multigraph: false,
  graph: {},
  nodes: [
    {
      id: 'benevolence',
    },
    {
      id: 'divine benevolence',
    },
    {
      id: 'real cause',
    },
    {
      id: 'true substance',
    },
    {
      id: 'physical substance',
    },
    {
      id: 'sensible ideas',
    },
    {
      id: 'true cause',
    },
    {
      id: 'divine agency',
    },
    {
      id: 'real power',
    },
    {
      id: 'general benevolence',
    },
    {
      id: 'true virtuous benevolence',
    },
    {
      id: 'true benevolence',
    },
    {
      id: 'divine reality',
    },
    {
      id: 'new simple idea',
    },
    {
      id: 'holy things evoke',
    },
    {
      id: 'true beauty',
    },
    {
      id: 'new spiritual sensation',
    },
    {
      id: 'simple ideas',
    },
    {
      id: 'human benevolence',
    },
    {
      id: 'spiritual beauty',
    },
    {
      id: 'benevolence mirrors',
    },
    {
      id: 'divine things',
    },
    {
      id: 'divine throne chariot',
    },
  ],
  edges: [
    {
      weight: 1,
      source: 'benevolence',
      target: 'divine benevolence',
    },
    {
      weight: 1,
      source: 'benevolence',
      target: 'general benevolence',
    },
    {
      weight: 1,
      source: 'benevolence',
      target: 'true virtuous benevolence',
    },
    {
      weight: 3,
      source: 'benevolence',
      target: 'true benevolence',
    },
    {
      weight: 2,
      source: 'benevolence',
      target: 'true beauty',
    },
    {
      weight: 1,
      source: 'benevolence',
      target: 'new simple idea',
    },
    {
      weight: 1,
      source: 'benevolence',
      target: 'human benevolence',
    },
    {
      weight: 1,
      source: 'benevolence',
      target: 'spiritual beauty',
    },
    {
      weight: 1,
      source: 'benevolence',
      target: 'benevolence mirrors',
    },
    {
      weight: 1,
      source: 'real cause',
      target: 'true substance',
    },
    {
      weight: 3,
      source: 'true substance',
      target: 'true cause',
    },
    {
      weight: 1,
      source: 'physical substance',
      target: 'sensible ideas',
    },
    {
      weight: 1,
      source: 'divine agency',
      target: 'real power',
    },
    {
      weight: 1,
      source: 'divine reality',
      target: 'new simple idea',
    },
    {
      weight: 1,
      source: 'new simple idea',
      target: 'simple ideas',
    },
    {
      weight: 1,
      source: 'holy things evoke',
      target: 'true beauty',
    },
    {
      weight: 1,
      source: 'new spiritual sensation',
      target: 'simple ideas',
    },
    {
      weight: 1,
      source: 'divine things',
      target: 'divine throne chariot',
    },
  ],
}
