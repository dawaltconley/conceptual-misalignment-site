import type { ReactNode } from 'react'
import { cn } from '@lib/cn'
import { Progress as ProgressRoot } from './ui/progress'

export interface ProgressProps {
  /** Current progress, in `max` units. Omit for an indeterminate bar. */
  value?: number
  /** What `value` counts up to. Default 100, i.e. `value` is a percentage. */
  max?: number
  /** Rendered above the bar, on the left. */
  label?: ReactNode
  /** Show the rounded percentage above the bar, on the right. */
  showValue?: boolean
  /** Classes for the label + bar stack. */
  className?: string
  /** Classes for the bar itself — height, color, etc. */
  barClassName?: string
}

/**
 * A determinate or indeterminate progress bar over `ui/progress` (Radix
 * Progress), which on its own only takes a 0-100 percentage.
 *
 * `value`/`max` are converted to that percentage, with `getValueLabel`
 * reporting the real numbers to assistive tech. Omitting `value` gives
 * Radix's indeterminate state, drawn as a sweeping partial bar: `ui/progress`
 * positions its indicator with an inline transform, and a CSS animation is
 * the one thing that outranks an inline style without `!important`.
 */
export default function Progress({
  value,
  max = 100,
  label,
  showValue = false,
  className,
  barClassName,
}: ProgressProps): JSX.Element {
  const indeterminate = value === undefined
  const percent =
    indeterminate || max <= 0
      ? 0
      : Math.min(100, Math.max(0, (value / max) * 100))

  return (
    <div className={cn('flex flex-col gap-1', className)}>
      {(label !== undefined || showValue) && (
        <div className="flex items-baseline justify-between gap-2 text-sm text-muted-foreground">
          <span>{label}</span>
          {showValue && (
            <span className="tabular-nums">
              {indeterminate ? '—' : `${Math.round(percent)}%`}
            </span>
          )}
        </div>
      )}
      <ProgressRoot
        value={indeterminate ? null : percent}
        getValueLabel={() => `${value} of ${max}`}
        className={cn(
          indeterminate &&
            '[&>*]:w-1/3 [&>*]:animate-progress-indeterminate [&>*]:transition-none',
          barClassName,
        )}
      />
    </div>
  )
}
