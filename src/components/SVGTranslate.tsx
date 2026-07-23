import type { ReactNode } from 'react'
import type { Point } from '@lib/graphs'

export interface SVGTranslateProps extends Partial<Point> {
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
