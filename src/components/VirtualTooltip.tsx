import type { ReactNode, RefObject } from 'react'
import { useMemo } from 'react'
import Tooltip from './Tooltip'

export interface VirtualTooltipProps {
  open: boolean
  /** Only fires for presses genuinely outside `containerRef` — see hazard note below. */
  onDismiss: () => void
  point: { x: number; y: number } | null
  containerRef: RefObject<Element | null>
  children: ReactNode
}

/**
 * Content-agnostic popup anchored to a synthesized point (e.g. canvas hit-test
 * position) rather than a real DOM node. Since the anchor is virtual, Base
 * UI's outside-press detection can't tell "outside" from "elsewhere on the
 * same canvas" (there's no real trigger element for it to register) — every
 * press while open is reported as outside-press. We only forward `onDismiss`
 * when the press actually landed outside `containerRef`, so a caller that
 * owns pointer/hit-test state for the whole surface (e.g.
 * `CanvasScatterPlot`) can safely pass its own gesture close handler through
 * without racing its own pointerdown handling for the same event.
 */
export default function VirtualTooltip({
  open,
  onDismiss,
  point,
  containerRef,
  children,
}: VirtualTooltipProps): JSX.Element {
  const anchor = useMemo(
    () =>
      point
        ? { getBoundingClientRect: () => new DOMRect(point.x, point.y, 0, 0) }
        : null,
    [point?.x, point?.y],
  )

  return (
    <Tooltip
      open={open}
      anchor={anchor}
      onDismiss={(event) => {
        if (!containerRef.current?.contains(event.target as Node)) onDismiss()
      }}
    >
      {children}
    </Tooltip>
  )
}
