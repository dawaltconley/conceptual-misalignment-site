import type { ReactNode } from 'react'

export interface SVGTranslateProps {
  x?: number
  y?: number
  children: ReactNode
}

export default function SVGTranslate({
  x = 0,
  y = 0,
  children,
}: SVGTranslateProps): ReactNode {
  if (!x && !y) return children
  return <g transform={`translate(${x},${y})`}>{children}</g>
}
