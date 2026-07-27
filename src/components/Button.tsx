import type { ReactNode, ComponentPropsWithoutRef } from 'react'
import clsx from 'clsx'

interface ButtonProps extends ComponentPropsWithoutRef<'button'> {
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
    <button
      className={clsx(
        'btn',
        isActive && 'btn--active',
        isPrimary && 'btn--primary',
        className,
      )}
      {...props}
    >
      {children}
    </button>
  )
}
