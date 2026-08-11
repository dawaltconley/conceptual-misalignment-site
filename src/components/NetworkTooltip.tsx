import type { PointerEvent, ReactNode, RefObject } from 'react'
import { useEffect, useRef } from 'react'
import useTooltipGesture from '@lib/browser/hooks/useTooltipGesture'
import Tooltip from './Tooltip'

export interface UseNetworkTooltipOptions {
  /** ms of mouse/pen hover before opening. */
  hoverDelay?: number
}

export interface UseNetworkTooltipResult {
  open: boolean
  close: () => void
  triggerProps: {
    onPointerEnter: (e: PointerEvent) => void
    onPointerLeave: (e: PointerEvent) => void
    onPointerDown: (e: PointerEvent) => void
  }
}

/**
 * Wires up tap/hold/hover for a single, real DOM-anchored trigger — e.g. a
 * `Network` graph node. Spread `triggerProps` onto the trigger element and
 * render the paired `NetworkTooltip` with the same `open`/`close`.
 *
 * Touch pointermove/up/cancel are tracked via `document`-level listeners
 * rather than the trigger's own React handlers, since an ancestor element
 * (e.g. a drag wrapper calling `setPointerCapture`) can redirect them away
 * from the trigger after pointerdown — captured events still bubble to
 * `document` regardless of which element captured them.
 */
export function useNetworkTooltip<T>(
  id: T,
  { hoverDelay = 300 }: UseNetworkTooltipOptions = {},
): UseNetworkTooltipResult {
  const gesture = useTooltipGesture<T>({ hoverDelay })
  const open = gesture.target === id
  const activePointerId = useRef<number | null>(null)
  const cleanup = useRef<() => void>(() => {})

  function onPointerDown(e: PointerEvent) {
    gesture.onPointerDown(id, { x: e.clientX, y: e.clientY }, e.pointerType)
    if (e.pointerType === 'mouse') return

    activePointerId.current = e.pointerId
    const onMove = (ev: globalThis.PointerEvent) => {
      if (ev.pointerId !== activePointerId.current) return
      gesture.onPointerMove({ x: ev.clientX, y: ev.clientY })
    }
    const onUp = (ev: globalThis.PointerEvent) => {
      if (ev.pointerId !== activePointerId.current) return
      gesture.onPointerUp()
      cleanup.current()
    }
    const onCancel = (ev: globalThis.PointerEvent) => {
      if (ev.pointerId !== activePointerId.current) return
      gesture.onPointerCancel()
      cleanup.current()
    }
    cleanup.current = () => {
      document.removeEventListener('pointermove', onMove)
      document.removeEventListener('pointerup', onUp)
      document.removeEventListener('pointercancel', onCancel)
      activePointerId.current = null
      cleanup.current = () => {}
    }
    document.addEventListener('pointermove', onMove)
    document.addEventListener('pointerup', onUp)
    document.addEventListener('pointercancel', onCancel)
  }

  useEffect(() => () => cleanup.current(), [])

  return {
    open,
    close: gesture.close,
    triggerProps: {
      onPointerEnter: (e) => e.pointerType === 'mouse' && gesture.onHover(id),
      onPointerLeave: (e) => e.pointerType === 'mouse' && gesture.onHover(null),
      onPointerDown,
    },
  }
}

export interface NetworkTooltipProps {
  open: boolean
  onDismiss: (event: Event) => void
  anchor: RefObject<Element | null>
  children: ReactNode
}

/** Content-agnostic popup for any `Network`-graph node — pairs with `useNetworkTooltip`. */
export default function NetworkTooltip({
  open,
  onDismiss,
  anchor,
  children,
}: NetworkTooltipProps): JSX.Element {
  return (
    <Tooltip open={open} onDismiss={onDismiss} anchor={anchor}>
      {children}
    </Tooltip>
  )
}
