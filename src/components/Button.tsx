import type { ReactNode } from 'react'
import {
  Button as BaseButton,
  type ButtonProps as BaseButtonProps,
} from '@base-ui/react/button'
import clsx from 'clsx'

interface ButtonProps extends BaseButtonProps {
  isPrimary?: boolean
  children: ReactNode
}

/**
 * A plain button. For a two-state button, use `Toggle` — it shares these
 * styles and carries the pressed state.
 */
export default function Button({
  children,
  className,
  isPrimary,
  ...props
}: ButtonProps): JSX.Element {
  return (
    <BaseButton
      className={clsx('btn', isPrimary && 'btn--primary', className)}
      {...props}
    >
      {children}
    </BaseButton>
  )
}
