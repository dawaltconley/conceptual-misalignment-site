import clsx from 'clsx'

export interface TickRangeProps {
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step?: number
  /**
   * Values to mark under the track. With `snap`, these are also the only values
   * the control can take — the marks are then a promise about where it stops.
   */
  ticks?: number[]
  /** Restrict `value` to the nearest tick. No effect without `ticks`. */
  snap?: boolean
  disabled?: boolean
  className?: string
  'aria-label'?: string
}

/**
 * A range input that can be pinned to a set of marked values.
 *
 * The scatter uses it for perplexity: the pipeline precomputes a few layouts, so
 * off-tick values are only reachable when the client is allowed to recompute.
 * Rather than swap in a different control for the two cases, the same slider
 * snaps to the marks or runs free — the ticks stay visible either way, so it is
 * clear which values cost nothing.
 */
export default function TickRange({
  value,
  onChange,
  min,
  max,
  step = 1,
  ticks,
  snap = false,
  disabled = false,
  className,
  'aria-label': ariaLabel,
}: TickRangeProps): JSX.Element {
  const marks = ticks?.filter((t) => t >= min && t <= max) ?? []
  const snapping = snap && marks.length > 0

  const percent = (v: number): number =>
    max === min ? 0 : ((v - min) / (max - min)) * 100

  return (
    <span className={clsx('range', className)}>
      <input
        className="range__input"
        type="range"
        min={min}
        max={max}
        // Snapping does its own quantizing, and the marks are not necessarily
        // evenly spaced, so the input itself stays fine-grained.
        step={step}
        value={value}
        disabled={disabled}
        aria-label={ariaLabel}
        onChange={(e) => {
          const next = Number(e.target.value)
          onChange(snapping ? nearest(marks, next) : next)
        }}
      />
      {marks.length > 0 && (
        <span className="range__ticks" aria-hidden="true">
          {marks.map((t) => (
            <span
              key={t}
              className="range__tick"
              style={{ left: `${percent(t)}%` }}
              {...(t === value ? { 'data-current': '' } : {})}
            />
          ))}
        </span>
      )}
    </span>
  )
}

/**
 * The closest of `values` to `target`, or `target` itself when there are none.
 * Exported because a caller that snaps has to agree with the control about where
 * a value lands — the scatter needs it to re-snap when recompute is turned off.
 */
export function nearest(values: number[], target: number): number {
  if (!values.length) return target
  return values.reduce((best, v) =>
    Math.abs(v - target) < Math.abs(best - target) ? v : best,
  )
}
