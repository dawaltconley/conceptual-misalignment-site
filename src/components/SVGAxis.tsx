import type { Point, Size } from '@lib/graphs'
import { useRef, useLayoutEffect } from 'react'
import * as d3 from 'd3'

type Orientation = 'left' | 'bottom'
type Scale = d3.ScaleLinear<number, number, never>

export interface GraphAxisProps extends Size {
  orientation: Orientation
  scale: Scale
  width: number
  height: number
  position?: Partial<Point>
}

export default function GraphAxis({
  orientation,
  scale,
  width,
  height,
  position = {},
}: GraphAxisProps): JSX.Element {
  const ref = useRef<SVGGElement>(null)
  const { x = 0, y = 0 } = position

  useLayoutEffect(() => {
    const host = d3.select(ref.current)
    const axisGenerator = getAxis(orientation, scale)
    const [start, end] = d3.extent(scale.range())
    if (start == null || end == null) {
      return
    }
    const pxPerTick = 80
    const tickCount = Math.ceil((end - start) / pxPerTick)
    axisGenerator.ticks(tickCount)

    let group = host.select<SVGGElement>('g')
    if (group.empty()) {
      group = host.append('g')
    }
    group.call(axisGenerator)
  }, [scale])

  return (
    <g
      ref={ref}
      width={width}
      height={height}
      transform={`translate(${x}, ${y})`}
    />
  )
}

function getAxis(
  orientation: Orientation,
  scale: Scale,
): d3.Axis<d3.NumberValue> {
  switch (orientation) {
    case 'left':
      return d3.axisLeft(scale)
    case 'bottom':
      return d3.axisBottom(scale)
  }
}
