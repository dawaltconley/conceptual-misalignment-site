export interface Size {
  width: number
  height: number
}

export interface Point {
  x: number
  y: number
}

export class Position implements Point, Size {
  width: number
  height: number
  x: number
  y: number

  constructor(opts: Point & Size) {
    this.width = opts.width
    this.height = opts.height
    this.x = opts.x
    this.y = opts.y
  }

  get size(): Size {
    const { width, height } = this
    return { width, height }
  }

  get pos(): Point {
    const { x, y } = this
    return { x, y }
  }
}

export function getDistSq(a: Point, b: Point): number {
  const dx = a.x - b.x
  const dy = a.y - b.y
  return dx * dx + dy * dy
}
