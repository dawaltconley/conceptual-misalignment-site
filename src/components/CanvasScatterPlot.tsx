import type { Dictionary } from '@build/cedict'
import {
  useState,
  useRef,
  useLayoutEffect,
  memo,
  useMemo,
  type MouseEvent,
  type PointerEvent,
} from 'react'
import * as d3 from 'd3'
import { Position, getDistSq } from '@lib/graphs'
import useSize from '@lib/browser/hooks/useSize'
import useTooltipGesture from '@lib/browser/hooks/useTooltipGesture'
import SVGAxis from './SVGAxis'
import SVGTranslate from './SVGTranslate'
import {
  Wrapper,
  type ScatterPoint,
  type ScatterPlotProps,
} from './ScatterPlot'
import VirtualTooltip from './VirtualTooltip'
import HanziDefinition from './HanziDefinition'
import clsx from 'clsx'

// --- layout (kept identical to ScatterPlot so the two are interchangeable) ---
const paddingX = 2
const paddingY = 2
const bottomAxisHeight = 30
const leftAxisWidth = 50
const rangeMultiplier = 1.05

const TEXT_THRESHOLD = 4
const DEFAULT_RADIUS = 3
const DEFAULT_COLOR = '#808080'
const DEFAULT_OPACITY = 0.7

export interface CanvasScatterPlotProps extends ScatterPlotProps {
  isHighlighted?: (p: ScatterPoint) => boolean
  dictionary?: Dictionary
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
  targets,
  isHighlighted = () => false,
  dictionary = {},
}: CanvasScatterPlotProps): JSX.Element {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const hoverCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const zoomRef = useRef<d3.ZoomBehavior<HTMLCanvasElement, unknown> | null>(
    null,
  )
  const size = useSize(containerRef)
  const { width = 0, height = 0 } = size || {}

  const [isInteractive, setIsInteractive] = useState(true)

  const gesture = useTooltipGesture<string>({
    hoverDelay: 0,
    holdDelay: 400,
    moveTolerance: 8,
  })
  // Read inside the zoom effect via a ref so pan-start can close a touch
  // tooltip without adding `gesture` to that effect's deps (which would tear
  // down/rebuild d3.zoom on every render).
  const gestureRef = useRef(gesture)
  gestureRef.current = gesture

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

  const xScale = useMemo(() => {
    const extent = d3.extent(points, (p) => p.x * rangeMultiplier) ?? [0, 1]
    return d3
      .scaleLinear()
      .domain(extent[0] === undefined ? [0, 1] : extent)
      .range([0, body.width])
  }, [points, body.width])

  const yScale = useMemo(() => {
    const extent = d3.extent(points, (p) => p.y * rangeMultiplier) ?? [0, 1]
    return d3
      .scaleLinear()
      .domain(extent[0] === undefined ? [0, 1] : extent)
      .range([body.height, 0])
  }, [points, body.height])

  const ready = width > 0 && height > 0

  // Semantic zoom: apply the pan/zoom transform to the SCALES, so point positions
  // spread/shift but marker radii stay constant. Axes and hit-testing use these.
  const [transform, setTransform] = useState<d3.ZoomTransform>(d3.zoomIdentity)
  const zx = useMemo(() => transform.rescaleX(xScale), [transform, xScale])
  const zy = useMemo(() => transform.rescaleY(yScale), [transform, yScale])
  const isZoomed = transform.k > 1

  // --- draw the point cloud on the canvas (repaints whenever points/size/zoom change) ---
  useLayoutEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !ready) return
    const ctx = canvas.getContext('2d', { alpha: true })
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr // also resets the transform to identity
    canvas.height = height * dpr
    ctx.scale(dpr, dpr)
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, width, height)
    ctx.translate(body.x, body.y) // draw in body-local coords (like the SVG body group)
    // Clip to the plot body so zoomed/panned points don't spill into the gutters.
    ctx.beginPath()
    ctx.rect(0, 0, body.width, body.height)
    ctx.clip()

    // Non-targets first (behind), then targets + labels on top.
    for (const p of points) {
      if (!p) continue
      if (targets.has(p.id)) continue
      const cx = zx(p.x)
      const cy = zy(p.y)
      const r = p.size ?? DEFAULT_RADIUS
      const alpha = p.opacity ?? DEFAULT_OPACITY

      ctx.globalAlpha = alpha
      ctx.beginPath()
      ctx.fillStyle = p.color || DEFAULT_COLOR
      ctx.arc(cx, cy, r, 0, 2 * Math.PI)
      ctx.fill()
      if (isHighlighted(p)) {
        ctx.strokeStyle = '#000000'
      } else {
        ctx.strokeStyle = p.color || DEFAULT_COLOR
        ctx.globalAlpha = alpha + (1 - alpha) * 0.2
      }
      ctx.stroke()
      if (transform.k > TEXT_THRESHOLD) {
        ctx.globalAlpha = 1
        ctx.font = '14px sans-serif'
        ctx.textBaseline = 'middle'
        ctx.fillStyle = '#111827'
        ctx.fillText(p.id, cx + 9, cy)
      }
    }

    ctx.globalAlpha = 1
    ctx.font = 'bold 14px sans-serif'
    ctx.textBaseline = 'middle'
    for (const p of points) {
      if (!targets.has(p.id)) continue
      const cx = zx(p.x)
      const cy = zy(p.y)
      const r = p.size ?? DEFAULT_RADIUS
      ctx.beginPath()
      ctx.fillStyle = p.color || DEFAULT_COLOR
      ctx.arc(cx, cy, r * 2, 0, 2 * Math.PI)
      ctx.fill()
      ctx.lineWidth = 1.5
      ctx.strokeStyle = '#111827'
      ctx.stroke()
      ctx.fillStyle = '#111827'
      ctx.fillText(p.id, cx + 9, cy)
    }
  }, [points, targets, width, height, isHighlighted, transform])

  // --- ctrl+wheel to zoom, drag to pan (bounded to the plot body) ---
  useLayoutEffect(() => {
    const canvas = hoverCanvasRef.current
    if (!canvas || !ready) return
    const zoom = d3
      .zoom<HTMLCanvasElement, unknown>()
      .scaleExtent([1, 8])
      .extent([
        [0, 0],
        [body.width, body.height],
      ])
      .translateExtent([
        [0, 0],
        [body.width, body.height],
      ])
      .on('zoom', (e: d3.D3ZoomEvent<HTMLCanvasElement, unknown>) =>
        setTransform(e.transform),
      )
      // A pinch/pan starting elsewhere would otherwise leave a touch-opened
      // tooltip visually detached from its (now-stale) anchor position. Only
      // close an *already-open* tooltip, though — d3-zoom fires 'start' on
      // every touchstart (even one that stays a stationary tap), so closing
      // unconditionally would kill the very touch that's trying to open one.
      .on('start', () => {
        if (gestureRef.current.target != null) gestureRef.current.close()
      })
    zoomRef.current = zoom

    const selection = d3.select(canvas)
    if (isInteractive) {
      selection.call(zoom)
      // Double-click resets to fit instead of d3's default zoom-in.
      selection.on('dblclick.zoom', null).on('dblclick', () => {
        selection.call(zoom.transform, d3.zoomIdentity)
      })
    }

    return () => {
      selection.on('.zoom', null).on('dblclick', null)
    }
  }, [width, height, body.width, body.height, isInteractive])

  // Reset zoom/pan when the point set changes (e.g. toggling PCA <-> t-SNE), so a
  // new layout always starts fit-to-view. Resetting via zoom.transform also clears
  // d3's internal node transform, not just our state, so the next gesture doesn't jump.
  const resetTimeout = useRef(0)
  useLayoutEffect(() => {
    const canvas = hoverCanvasRef.current
    const zoom = zoomRef.current
    if (!canvas || !zoom) return
    d3.select(canvas).call(zoom.transform, d3.zoomIdentity)

    window.clearTimeout(resetTimeout.current)
    setIsInteractive(false)
    resetTimeout.current = window.setTimeout(() => setIsInteractive(true), 300)
  }, [points])

  const delaunay = useMemo(
    () =>
      isInteractive
        ? d3.Delaunay.from(
            points,
            (p) => xScale(p.x),
            (p) => yScale(p.y),
          )
        : null,
    [points, xScale, yScale, isInteractive],
  )

  const [tooltip, setTooltip] = useState<ScatterPoint | null>(null)
  const hanziDefinition = tooltip && dictionary[tooltip.id]

  // Delaunay lookup shared by mouse hover and touch tap/hold, so the target-node
  // exclusion and distance cutoff below apply identically to both input paths.
  // Lives in un-zoomed scale-pixel space, so invert the zoom first.
  const hitTest = (clientX: number, clientY: number): ScatterPoint | null => {
    const canvas = hoverCanvasRef.current
    if (!canvas || !ready || !delaunay) return null
    const bounds = canvas.getBoundingClientRect()
    const x = clientX - bounds.left - body.x
    const y = clientY - bounds.top - body.y

    const [ix, iy] = transform.invert([x, y])
    const index = delaunay.find(ix, iy)
    const p: ScatterPoint | undefined = points[index]
    if (!p || targets.has(p.id)) return null

    const px = zx(p.x)
    const py = zy(p.y)
    if (getDistSq({ x: px, y: py }, { x, y }) > 1200) return null

    return p
  }

  // Viewport-absolute position for a hit point, for the tooltip's virtual anchor.
  const anchorPoint = (p: ScatterPoint): ScatterPoint => {
    const bounds = hoverCanvasRef.current?.getBoundingClientRect()
    return {
      ...p,
      x: zx(p.x) + body.x + (bounds?.left ?? 0),
      y: zy(p.y) + body.y + (bounds?.top ?? 0),
    }
  }

  const handleHover = (e: MouseEvent<HTMLCanvasElement> | null) => {
    const canvas = hoverCanvasRef.current
    if (!canvas || !ready) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr // also resets the transform to identity
    canvas.height = height * dpr
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, width, height)
    ctx.translate(body.x, body.y) // draw in body-local coords (like the SVG body group)
    ctx.beginPath()
    ctx.rect(0, 0, body.width, body.height)
    ctx.clip()

    // Clear on leave, and don't hover while panning (drag holds the button down).
    if (!e || e.buttons) {
      setTooltip(null)
      gesture.onHover(null)
      return
    }

    const p = hitTest(e.clientX, e.clientY)
    gesture.onHover(p?.id ?? null)
    if (!p) {
      setTooltip(null)
      return
    }

    setTooltip(anchorPoint(p))

    const px = zx(p.x)
    const py = zy(p.y)
    const r = p.size ?? DEFAULT_RADIUS
    ctx.beginPath()
    ctx.arc(px, py, r * 1.1, 0, 2 * Math.PI)
    ctx.lineWidth = 1.5
    ctx.strokeStyle = '#000000'
    ctx.stroke()
    ctx.fillStyle = p.color || DEFAULT_COLOR
    ctx.fill()
  }

  // Touch/pen only — mouse keeps using handleHover above. Hit-testing eagerly on
  // pointerdown lets the canvas ring/tooltip anchor track the press immediately;
  // whether the popup actually opens is decided by `gesture` (tap, hold-still,
  // or a pan that never resolves into either).
  const handlePress = (e: PointerEvent<HTMLCanvasElement>) => {
    if (e.pointerType === 'mouse') return
    const p = isInteractive ? hitTest(e.clientX, e.clientY) : null
    setTooltip(p ? anchorPoint(p) : null)
    gesture.onPointerDown(
      p?.id ?? null,
      { x: e.clientX, y: e.clientY },
      e.pointerType,
    )
  }

  const handlePressMove = (e: PointerEvent<HTMLCanvasElement>) => {
    if (e.pointerType === 'mouse') return
    gesture.onPointerMove({ x: e.clientX, y: e.clientY })
  }

  const handlePressEnd = (e: PointerEvent<HTMLCanvasElement>) => {
    if (e.pointerType === 'mouse') return
    gesture.onPointerUp()
  }

  const handlePressCancel = (e: PointerEvent<HTMLCanvasElement>) => {
    if (e.pointerType === 'mouse') return
    gesture.onPointerCancel()
  }

  return (
    <Wrapper ref={containerRef}>
      <canvas
        ref={canvasRef}
        className="absolute inset-0"
        style={{ width, height }}
      />
      <canvas
        ref={hoverCanvasRef}
        className={clsx(
          'absolute inset-0',
          isZoomed && 'cursor-grab active:cursor-grabbing',
          tooltip && 'cursor-pointer',
        )}
        style={{ width, height }}
        onMouseMove={(e) => isInteractive && handleHover(e)}
        onMouseOut={() => handleHover(null)}
        onPointerDown={handlePress}
        onPointerMove={handlePressMove}
        onPointerUp={handlePressEnd}
        onPointerCancel={handlePressCancel}
      />
      {ready && (
        <svg
          width={width}
          height={height}
          className="pointer-events-none absolute inset-0"
          viewBox={`0 0 ${width} ${height}`}
        >
          <SVGTranslate {...leftAxis.pos}>
            <SVGAxis orientation="left" scale={zy} {...leftAxis.size} />
          </SVGTranslate>
          <SVGTranslate {...bottomAxis.pos}>
            <SVGAxis orientation="bottom" scale={zx} {...bottomAxis.size} />
          </SVGTranslate>
        </svg>
      )}
      <VirtualTooltip
        open={
          gesture.target != null &&
          tooltip != null &&
          gesture.target === tooltip.id
        }
        onDismiss={gesture.close}
        point={tooltip}
        containerRef={hoverCanvasRef}
      >
        {hanziDefinition ? (
          <HanziDefinition entry={hanziDefinition} />
        ) : (
          <div className="rounded-sm bg-white p-2 text-sm shadow-xl ring-1 ring-gray-200">
            {tooltip?.id}
          </div>
        )}
      </VirtualTooltip>
    </Wrapper>
  )
}

export default memo(CanvasScatterPlot)
