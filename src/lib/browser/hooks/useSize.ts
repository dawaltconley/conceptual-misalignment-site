import { useState, useLayoutEffect, type RefObject } from 'react'
import useResizeObserver from '@react-hook/resize-observer'

export default function useSize(
  target: RefObject<Element>,
): DOMRect | undefined {
  const [size, setSize] = useState<DOMRect>()
  useLayoutEffect(() => {
    if (target.current) {
      setSize(target.current.getBoundingClientRect())
    }
  }, [target])
  useResizeObserver(target, (entry) => setSize(entry.contentRect))
  return size
}
