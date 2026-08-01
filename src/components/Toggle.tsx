import type { ReactNode } from 'react'
import { Toggle as BaseToggle } from '@base-ui/react/toggle'
import clsx from 'clsx'

export interface ToggleProps extends Omit<BaseToggle.Props, 'className'> {
  /** Narrowed from Base UI's `string | ((state) => string)` so `clsx` can merge it. */
  className?: string
  children: ReactNode
}

/**
 * A two-state button over Base UI's Toggle, sharing `Button`'s `.btn` styles so
 * the two read as one control in a row. Base UI marks the on state with
 * `[data-pressed]`, which `.btn[data-pressed]` styles (see `tailwind.scss`).
 *
 * Controlled via `pressed` / `onPressedChange`; `defaultPressed` for the
 * uncontrolled case.
 */
export default function Toggle({
  children,
  className,
  ...props
}: ToggleProps): JSX.Element {
  return (
    <BaseToggle className={clsx('btn', className)} {...props}>
      {children}
    </BaseToggle>
  )
}
