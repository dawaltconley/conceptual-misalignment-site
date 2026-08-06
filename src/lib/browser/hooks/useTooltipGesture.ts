import { useEffect, useRef, useState } from 'react'

export interface UseTooltipGestureOptions {
  /** ms of continuous mouse/pen hover on a target before it opens. 0 = open synchronously. */
  hoverDelay?: number
  /** ms a stationary touch press must be held before it opens (mirrors hover-to-peek). */
  holdDelay?: number
  /** px of pointer movement during a hold that cancels it (treated as the start of a pan). */
  moveTolerance?: number
}

export interface UseTooltipGestureResult<T> {
  /** Currently revealed target, or null. */
  target: T | null
  /** Mouse/pen hover path — call with the hovered target (or null) on enter/leave/hit-test. */
  onHover: (target: T | null) => void
  /** Any-pointer-type pointerdown. Mouse just clears hover state; touch/pen runs tap/hold. */
  onPointerDown: (
    target: T | null,
    point: { x: number; y: number },
    pointerType: string,
  ) => void
  /** Touch/pen only; cancels a pending hold if movement exceeds moveTolerance. */
  onPointerMove: (point: { x: number; y: number }) => void
  /** Touch/pen only; opens immediately if released before the hold resolves (a tap). */
  onPointerUp: () => void
  /** Touch/pen only; abandons the gesture without opening. */
  onPointerCancel: () => void
  /** Unconditional close, for external dismiss sources (outside-press, unmount, …). */
  close: () => void
}

const DEFAULT_HOVER_DELAY = 0
const DEFAULT_HOLD_DELAY = 400
const DEFAULT_MOVE_TOLERANCE = 64

type Phase =
  | 'idle'
  | 'hover-pending'
  | 'open-hover'
  | 'press-pending'
  | 'open-press'

export default function useTooltipGesture<T>({
  hoverDelay = DEFAULT_HOVER_DELAY,
  holdDelay = DEFAULT_HOLD_DELAY,
  moveTolerance = DEFAULT_MOVE_TOLERANCE,
}: UseTooltipGestureOptions = {}): UseTooltipGestureResult<T> {
  const [target, setTarget] = useState<T | null>(null)
  const phase = useRef<Phase>('idle')
  const pendingTarget = useRef<T | null>(null)
  const startPoint = useRef<{ x: number; y: number } | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  function clearTimer() {
    if (timer.current !== null) {
      clearTimeout(timer.current)
      timer.current = null
    }
  }

  useEffect(() => clearTimer, [])

  function close() {
    clearTimer()
    phase.current = 'idle'
    pendingTarget.current = null
    startPoint.current = null
    setTarget(null)
  }

  function onHover(next: T | null) {
    if (phase.current === 'press-pending' || phase.current === 'open-press')
      return

    if (next === null) {
      clearTimer()
      phase.current = 'idle'
      if (target !== null) setTarget(null)
      return
    }

    if (phase.current === 'open-hover' && target === next) return

    clearTimer()
    pendingTarget.current = next
    if (hoverDelay <= 0) {
      phase.current = 'open-hover'
      setTarget(next)
      return
    }
    phase.current = 'hover-pending'
    timer.current = setTimeout(() => {
      phase.current = 'open-hover'
      setTarget(pendingTarget.current)
    }, hoverDelay)
  }

  function onPointerDown(
    next: T | null,
    point: { x: number; y: number },
    pointerType: string,
  ) {
    if (pointerType === 'mouse') {
      clearTimer()
      if (phase.current === 'hover-pending' || phase.current === 'open-hover') {
        phase.current = 'idle'
        setTarget(null)
      }
      return
    }

    if (phase.current === 'open-press' && target === next) {
      close()
      return
    }

    clearTimer()
    if (next === null) {
      phase.current = 'idle'
      pendingTarget.current = null
      startPoint.current = null
      if (target !== null) setTarget(null)
      return
    }

    pendingTarget.current = next
    startPoint.current = point
    phase.current = 'press-pending'
    timer.current = setTimeout(() => {
      phase.current = 'open-press'
      setTarget(pendingTarget.current)
    }, holdDelay)
  }

  function onPointerMove(point: { x: number; y: number }) {
    if (phase.current !== 'press-pending' || !startPoint.current) return
    const dx = point.x - startPoint.current.x
    const dy = point.y - startPoint.current.y
    if (dx * dx + dy * dy > moveTolerance * moveTolerance) {
      clearTimer()
      phase.current = 'idle'
      pendingTarget.current = null
      startPoint.current = null
    }
  }

  function onPointerUp() {
    if (phase.current !== 'press-pending') return
    clearTimer()
    phase.current = 'open-press'
    setTarget(pendingTarget.current)
  }

  function onPointerCancel() {
    if (phase.current === 'press-pending' || phase.current === 'open-press')
      close()
  }

  return {
    target,
    onHover,
    onPointerDown,
    onPointerMove,
    onPointerUp,
    onPointerCancel,
    close,
  }
}
