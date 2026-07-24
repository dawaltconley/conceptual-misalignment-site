import { useRef, useLayoutEffect, memo } from 'react'
import * as d3 from 'd3'
import { Position } from '@lib/graphs'
import useSize from '@lib/browser/hooks/useSize'
import SVGAxis from './SVGAxis'
import SVGTranslate from './SVGTranslate'
import { Wrapper, type ScatterPlotProps } from './ScatterPlot'

export interface CanvasScatterPlotProps extends ScatterPlotProps {
  highlightedCommunities?: Set<number>
}

/**
 * Canvas variant of `ScatterPlot` — identical props, drop-in for `EmbeddingScatter`.
 *
 * The point cloud is drawn imperatively on a `<canvas>` (fast to repaint every
 * t-SNE frame, unlike hundreds of React-managed SVG nodes); the axes stay SVG,
 * reusing `SVGAxis`, overlaid in the same layout — the `Network.tsx` canvas +
 * overlay pattern. Trade-off vs. the SVG version: no per-point `<title>` hover
 * (would need canvas hit-testing).
 */
function CanvasScatterPlot({
  points,
  getColor,
  highlightedCommunities = new Set(),
}: CanvasScatterPlotProps): JSX.Element {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const size = useSize(containerRef)
  const { width = 0, height = 0 } = size || {}

  // --- layout (kept identical to ScatterPlot so the two are interchangeable) ---
  const paddingX = 2
  const paddingY = 2
  const bottomAxisHeight = 30
  const leftAxisWidth = 50
  const rangeMultiplier = 1.05

  const body = new Position({
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

  // --- draw the point cloud on the canvas (repaints whenever points/size change) ---
  useLayoutEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !ready) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr // also resets the transform to identity
    canvas.height = height * dpr
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, width, height)
    ctx.translate(body.x, body.y) // draw in body-local coords (like the SVG body group)

    // Non-targets first (behind), then targets + labels on top.
    ctx.globalAlpha = 0.7
    for (const p of points) {
      if (p.target) continue
      ctx.beginPath()
      ctx.fillStyle = getColor(p.community)
      ctx.arc(xScale(p.x), yScale(p.y), 3, 0, 2 * Math.PI)
      ctx.fill()
      if (highlightedCommunities.has(p.community)) {
        ctx.strokeStyle = '#000000'
        ctx.stroke()
      }
    }

    ctx.globalAlpha = 1
    ctx.font = 'bold 14px sans-serif'
    ctx.textBaseline = 'middle'
    for (const p of points) {
      if (!p.target) continue
      const cx = xScale(p.x)
      const cy = yScale(p.y)
      ctx.beginPath()
      ctx.fillStyle = getColor(p.community)
      ctx.arc(cx, cy, 7, 0, 2 * Math.PI)
      ctx.fill()
      ctx.lineWidth = 1.5
      ctx.strokeStyle = '#111827'
      ctx.stroke()
      ctx.fillStyle = '#111827'
      ctx.fillText(p.id, cx + 9, cy)
    }
  }, [points, width, height, highlightedCommunities])

  return (
    <Wrapper ref={containerRef}>
      <canvas
        ref={canvasRef}
        className="absolute inset-0"
        style={{ width, height }}
      />
      {ready && (
        <svg
          width={width}
          height={height}
          className="pointer-events-none absolute inset-0"
          viewBox={`0 0 ${width} ${height}`}
        >
          <SVGTranslate {...leftAxis.pos}>
            <SVGAxis orientation="left" scale={yScale} {...leftAxis.size} />
          </SVGTranslate>
          <SVGTranslate {...bottomAxis.pos}>
            <SVGAxis orientation="bottom" scale={xScale} {...bottomAxis.size} />
          </SVGTranslate>
        </svg>
      )}
    </Wrapper>
  )
}

export default memo(CanvasScatterPlot)
