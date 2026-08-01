import type { ReactNode } from 'react'
import { Progress as BaseProgress } from '@base-ui/react/progress'
import clsx from 'clsx'

export interface ProgressProps {
  /** Current progress, between `min` and `max`. Omit for an indeterminate bar. */
  value?: number
  min?: number
  /** What `value` counts up to. Default 100, i.e. `value` is a percentage. */
  max?: number
  /** Rendered above the bar, on the left. */
  label?: ReactNode
  /** Show a percentage readout above the bar, on the right. */
  showValue?: boolean
  /** Classes for the label + bar stack. */
  className?: string
  /** Classes for the track. */
  trackClassName?: string
}

/**
 * A determinate or indeterminate progress bar over Base UI's Progress.
 *
 * Omitting `value` gives the indeterminate state (Base UI's `value={null}`),
 * drawn as a sweeping partial bar — Base UI leaves the indicator's width unset
 * in that state, so the animation is pure CSS with nothing inline to fight.
 */
export default function Progress({
  value,
  min = 0,
  max = 100,
  label,
  showValue = false,
  className,
  trackClassName,
}: ProgressProps): JSX.Element {
  return (
    <BaseProgress.Root
      // Base UI reads `null` as indeterminate; `undefined` would fall back to
      // its own default rather than being passed through.
      value={value ?? null}
      min={min}
      max={max}
      className={clsx('flex flex-col gap-1', className)}
    >
      {(label !== undefined || showValue) && (
        <div className="flex items-baseline justify-between gap-2 text-sm text-gray-500">
          <BaseProgress.Label>{label}</BaseProgress.Label>
          {showValue && (
            // Base UI's Value renders the raw number; a percentage of the
            // min–max span is the more useful readout beside a unit label.
            <BaseProgress.Value className="tabular-nums">
              {(_, current) =>
                current === null
                  ? '—'
                  : `${Math.round(((current - min) / (max - min)) * 100)}%`
              }
            </BaseProgress.Value>
          )}
        </div>
      )}
      <BaseProgress.Track className={clsx('progress__track', trackClassName)}>
        <BaseProgress.Indicator className="progress__indicator" />
      </BaseProgress.Track>
    </BaseProgress.Root>
  )
}
