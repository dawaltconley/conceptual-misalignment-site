import { useState, useEffect, useRef } from 'react'

export type ScrollPosition = 'top' | 'middle' | 'bottom' | null

export interface UseScrollPositionOpts {
  threshold?: number
}

export default function useScrollPosition<T extends Element>({
  threshold = 0,
}: UseScrollPositionOpts) {
  const [position, setPosition] = useState<ScrollPosition>(null)
  const ref = useRef<T>(null)

  useEffect(() => {
    const e = ref.current
    if (!e) return
    const scrollListener = () => {
      setPosition(() => {
        if (e.scrollTop <= threshold) {
          return 'top'
        } else if (e.scrollHeight - e.scrollTop - e.clientHeight <= threshold) {
          return 'bottom'
        } else {
          return 'middle'
        }
      })
    }
    scrollListener() // fire once to capture initial state
    e.addEventListener('scroll', scrollListener, { passive: true })
    return () => {
      e.removeEventListener('scroll', scrollListener)
    }
  }, [])

  return { ref, position }
}
