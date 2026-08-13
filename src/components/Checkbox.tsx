import type { ReactNode } from 'react'
import { Checkbox as BaseCheckbox } from '@base-ui/react/checkbox'
import clsx from 'clsx'
import Icon from './Icon'
import { faCheck } from '@fortawesome/pro-regular-svg-icons/faCheck'

export interface CheckboxProps extends Omit<
  BaseCheckbox.Root.Props,
  'className' | 'render'
> {
  /** Narrowed from Base UI's `string | ((state) => string)` so `clsx` can merge it. */
  className?: string
  /** Rendered beside the box, and labels it. */
  children?: ReactNode
  /** Classes for the wrapping label. */
  labelClassName?: string
}

/**
 * A labelled checkbox over Base UI's Checkbox, using the shared `.checkbox`
 * component class (Base UI marks the on state with `[data-checked]`, the same
 * way `Toggle` uses `[data-pressed]`).
 *
 * Controlled via `checked` / `onCheckedChange`; `defaultChecked` otherwise.
 */
export default function Checkbox({
  children,
  className,
  labelClassName,
  ...props
}: CheckboxProps): JSX.Element {
  return (
    <label className={clsx('flex items-center gap-2', labelClassName)}>
      <BaseCheckbox.Root className={clsx('checkbox', className)} {...props}>
        <BaseCheckbox.Indicator>
          <Icon icon={faCheck} width="0.625em" height="0.625em" />
        </BaseCheckbox.Indicator>
      </BaseCheckbox.Root>
      {children}
    </label>
  )
}
