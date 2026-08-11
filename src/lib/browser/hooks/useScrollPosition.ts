import { useState, useEffect, useRef } from 'react'
import { useResizeObserver } from 'use-resize-observer'

export type ScrollPosition = 'top' | 'middle' | 'bottom' | null

export interface UseScrollPositionOpts {
  threshold?: number
}

export default function useScrollPosition<T extends Element>({
  threshold = 0,
}: UseScrollPositionOpts) {
  const [position, setPosition] = useState<ScrollPosition>(null)
  const ref = useRef<T>(null)

  useResizeObserver<T>({
    ref,
    onResize: ({ height }) => {
      const el = ref.current
      if (!el || !height) return
      console.log('resize', getScrollPosition(el, threshold))
      setPosition(() => getScrollPosition(el, threshold))
    },
  })

  useEffect(() => {
    const e = ref.current
    if (!e) return
    const scrollListener = () => {
      setPosition(() => getScrollPosition(e, threshold))
    }
    scrollListener() // fire once to capture initial state
    e.addEventListener('scroll', scrollListener, { passive: true })
    return () => {
      e.removeEventListener('scroll', scrollListener)
    }
  }, [threshold])

  return { ref, position }
}

function getScrollPosition(e: Element, threshold = 0): ScrollPosition {
  if (e.scrollHeight <= e.clientHeight) {
    return null
  }
  if (e.scrollTop <= threshold) {
    return 'top'
  } else if (e.scrollHeight - e.scrollTop - e.clientHeight <= threshold) {
    return 'bottom'
  } else {
    return 'middle'
  }
}
