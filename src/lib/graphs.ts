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
