import type { ReactNode } from 'react'
import {
  Button as BaseButton,
  type ButtonProps as BaseButtonProps,
} from '@base-ui/react/button'
import clsx from 'clsx'

interface ButtonProps extends BaseButtonProps {
  isPrimary?: boolean
  isActive?: boolean
  children: ReactNode
}

export default function Button({
  children,
  className,
  isPrimary,
  isActive,
  ...props
}: ButtonProps): JSX.Element {
  return (
    <BaseButton
      className={clsx(
        'btn',
        isActive && 'btn--active',
        isPrimary && 'btn--primary',
        className,
      )}
      {...props}
    >
      {children}
    </BaseButton>
  )
}
