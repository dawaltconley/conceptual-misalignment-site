import { useRef, forwardRef, memo, type ReactNode } from 'react'
import * as d3 from 'd3'
import { Position, type Point } from '@lib/graphs'
import useSize from '@lib/browser/hooks/useSize'
import SVGAxis from './SVGAxis'
import SVGTranslate from './SVGTranslate'

export interface ScatterPoint extends Point {
  id: string
  x: number
  y: number
  color?: string
  size?: number
  opacity?: number
  data?: Record<string, unknown>
}

export interface ScatterPlotProps {
  points: ScatterPoint[]
  targets: Set<string>
}

/**
 * Pure 2-D scatter renderer (the `Network.tsx` of scatter plots): given points
 * with fixed x/y, draws an SVG scatter coloured by community with targets
 * emphasised. No data fetching and no projection math — the wrapper owns those.
 */
function ScatterPlot({ points, targets }: ScatterPlotProps): JSX.Element {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const size = useSize(containerRef)
  const { width = 0, height = 0 } = size || {}

  const paddingX = 2
  const paddingY = 2
  const bottomAxisHeight = 30
  const leftAxisWidth = 50
  const rangeMultiplier = 1.05

  // Body origin (top-left of the plot area). The left-axis group shares this
  // origin — d3.axisLeft draws its spine at local x=0 (the body's left edge) and
  // its tick labels to the left, into the reserved gutter. The bottom-axis group
  // sits directly under the body.
  const body = new Position({
    // Plot body = full size minus outer padding and the axis gutters (left gutter
    // holds the y tick labels, bottom gutter holds the x tick labels).
    width: Math.max(width - leftAxisWidth - 2 * paddingX, 0),
    height: Math.max(height - bottomAxisHeight - 2 * paddingY, 0),
    x: paddingX + leftAxisWidth,
    y: paddingY,
  })
  const leftAxis = new Position({
    width: leftAxisWidth,
    height: body.height,
    x: body.x,
    y: body.y,
  })
  const bottomAxis = new Position({
    x: body.x,
    y: body.y + body.height,
    width: body.width,
    height: bottomAxisHeight,
  })

  // Scales map data into the body's LOCAL coordinate space, so the points (drawn
  // in the body group) and the axes (drawn in their own translated groups that
  // share the body's edges) line up exactly.
  const xExtent = d3.extent(points, (p) => p.x * rangeMultiplier) ?? [0, 1]
  const yExtent = d3.extent(points, (p) => p.y * rangeMultiplier) ?? [0, 1]
  const xScale = d3
    .scaleLinear()
    .domain(xExtent[0] === undefined ? [0, 1] : xExtent)
    .range([0, body.width])
  const yScale = d3
    .scaleLinear()
    .domain(yExtent[0] === undefined ? [0, 1] : yExtent)
    .range([body.height, 0])

  const ready = width > 0 && height > 0

  return (
    <Wrapper ref={containerRef}>
      <svg
        width={width}
        height={height}
        className="absolute inset-0"
        viewBox={`0 0 ${width} ${height}`}
      >
        <SVGTranslate {...body.pos}>
          {ready &&
            points.map((p) => {
              const cx = xScale(p.x)
              const cy = yScale(p.y)
              const fill = p.color || '#808080'
              const isTarget = targets.has(p.id)
              return (
                <g key={p.id}>
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isTarget ? 7 : 3}
                    fill={fill}
                    stroke={isTarget ? '#111827' : 'none'}
                    strokeWidth={isTarget ? 1.5 : 0}
                    opacity={isTarget ? 1 : 0.7}
                    data-community={p.data?.community?.toString()}
                  >
                    <title>{p.id}</title>
                  </circle>
                  {isTarget && (
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
        </SVGTranslate>
        {ready && (
          <>
            <SVGTranslate {...leftAxis.pos}>
              <SVGAxis orientation="left" scale={yScale} {...leftAxis.size} />
            </SVGTranslate>
            <SVGTranslate {...bottomAxis.pos}>
              <SVGAxis
                orientation="bottom"
                scale={xScale}
                {...bottomAxis.size}
              />
            </SVGTranslate>
          </>
        )}
      </svg>
    </Wrapper>
  )
}

export default memo(ScatterPlot)

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
      className="relative aspect-square min-h-96 w-full touch-manipulation overflow-hidden"
    >
      {children}
    </div>
  ),
)
