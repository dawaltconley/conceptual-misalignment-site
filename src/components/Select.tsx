import { useId, type ReactNode } from 'react'
import { cn } from '@lib/cn'
import {
  Select as SelectRoot,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select'

export interface SelectOption {
  value: string
  /** What the option reads as; defaults to `value`. */
  label?: ReactNode
  disabled?: boolean
}

export interface SelectProps {
  value: string
  /** Bare strings are treated as `{ value }`. */
  options: readonly (string | SelectOption)[]
  onChange: (value: string) => void
  /** Rendered beside the trigger and wired up as its accessible name. */
  label?: ReactNode
  placeholder?: string
  disabled?: boolean
  name?: string
  /** Classes for the label + trigger row. */
  className?: string
  /** Classes for the trigger itself — width, type size, etc. */
  triggerClassName?: string
}

/**
 * A single-choice dropdown over `ui/select` (Radix Select). Unlike a native
 * `<select>` the trigger is a button, so it can't be wrapped in a `<label>`;
 * the label is a sibling `<span>` referenced by `aria-labelledby` instead.
 */
export default function Select({
  value,
  options,
  onChange,
  label,
  placeholder,
  disabled,
  name,
  className,
  triggerClassName,
}: SelectProps): JSX.Element {
  const labelId = useId()
  const items = options.map<SelectOption>((o) =>
    typeof o === 'string' ? { value: o } : o,
  )

  return (
    <div className={cn('flex items-center gap-2', className)}>
      {label !== undefined && (
        <span id={labelId} className="text-sm text-muted-foreground">
          {label}
        </span>
      )}
      <SelectRoot
        value={value}
        onValueChange={onChange}
        disabled={disabled}
        name={name}
      >
        <SelectTrigger
          aria-labelledby={label === undefined ? undefined : labelId}
          className={cn('w-auto', triggerClassName)}
        >
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {items.map((o) => (
            <SelectItem key={o.value} value={o.value} disabled={o.disabled}>
              {o.label ?? o.value}
            </SelectItem>
          ))}
        </SelectContent>
      </SelectRoot>
    </div>
  )
}
