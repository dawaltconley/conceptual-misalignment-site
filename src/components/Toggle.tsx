import type { ComponentPropsWithoutRef, ReactNode } from 'react'
import { Toggle as ToggleRoot } from './ui/toggle'

type ToggleRootProps = ComponentPropsWithoutRef<typeof ToggleRoot>

export interface ToggleProps extends ToggleRootProps {
  /**
   * Whether the toggle reads as on. Controlled — pair with `onPressedChange`.
   * Radix also supports `defaultPressed` for the uncontrolled case.
   */
  pressed?: boolean
  onPressedChange?: (pressed: boolean) => void
  children: ReactNode
}

/**
 * A two-state button over `ui/toggle` (Radix Toggle). Defaults to the
 * `outline` variant, since these sit in rows beside outline Buttons; the
 * borderless `default` variant is still available via `variant`.
 *
 * For a row of *mutually exclusive* options this is one Toggle per option
 * with `pressed` derived from the current value. A Radix ToggleGroup would
 * model that better and is worth revisiting.
 */
export default function Toggle({
  variant = 'outline',
  children,
  ...props
}: ToggleProps): JSX.Element {
  return (
    <ToggleRoot variant={variant} {...props}>
      {children}
    </ToggleRoot>
  )
}
