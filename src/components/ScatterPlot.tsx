import { useRef, forwardRef, type ReactNode } from 'react'
import * as d3 from 'd3'
import useSize from '@lib/browser/hooks/useSize'

export interface ScatterPoint {
  id: string
  x: number
  y: number
  community: number
  target: boolean
}

export interface ScatterPlotProps {
  points: ScatterPoint[]
}

const NO_COMMUNITY_COLOR = '#cbd5e1' // grey for isolates (community -1)
const MARGIN = 24

/**
 * Pure 2-D scatter renderer (the `Network.tsx` of scatter plots): given points
 * with fixed x/y, draws an SVG scatter coloured by community with targets
 * emphasised. No data fetching and no projection math — the wrapper owns those.
 */
export default function ScatterPlot({ points }: ScatterPlotProps): JSX.Element {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const size = useSize(containerRef)
  const { width = 0, height = 0 } = size || {}

  const xExtent = (d3.extent(points, (p) => p.x) as [number, number]) ?? [0, 1]
  const yExtent = (d3.extent(points, (p) => p.y) as [number, number]) ?? [0, 1]
  const xScale = d3
    .scaleLinear()
    .domain(xExtent[0] === undefined ? [0, 1] : xExtent)
    .range([MARGIN, Math.max(width - MARGIN, MARGIN)])
  const yScale = d3
    .scaleLinear()
    .domain(yExtent[0] === undefined ? [0, 1] : yExtent)
    .range([Math.max(height - MARGIN, MARGIN), MARGIN])

  const color = d3.scaleOrdinal<number, string>(d3.schemeTableau10)

  return (
    <Wrapper ref={containerRef}>
      <svg width={width} height={height} className="absolute inset-0">
        {width > 0 &&
          points.map((p) => {
            const cx = xScale(p.x)
            const cy = yScale(p.y)
            const fill = p.community < 0 ? NO_COMMUNITY_COLOR : color(p.community)
            return (
              <g key={p.id}>
                <circle
                  cx={cx}
                  cy={cy}
                  r={p.target ? 7 : 3}
                  fill={fill}
                  stroke={p.target ? '#111827' : 'none'}
                  strokeWidth={p.target ? 1.5 : 0}
                  opacity={p.target ? 1 : 0.7}
                >
                  <title>{p.id}</title>
                </circle>
                {p.target && (
                  <text
                    x={cx + 9}
                    y={cy + 4}
                    className="fill-gray-900 text-sm font-bold"
                    style={{ pointerEvents: 'none' }}
                  >
                    {p.id}
                  </text>
                )}
              </g>
            )
          })}
      </svg>
    </Wrapper>
  )
}

export const ScatterSkeleton = (): JSX.Element => (
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
      className="relative aspect-square min-h-96 w-full overflow-hidden rounded ring-1 ring-gray-200"
    >
      {children}
    </div>
  ),
)
