import type { ReactNode } from 'react'
import { Select as BaseSelect } from '@base-ui/react/select'
import clsx from 'clsx'
import Icon from './Icon'
import { faChevronDown } from '@fortawesome/pro-regular-svg-icons/faChevronDown'
import { faCheck } from '@fortawesome/pro-regular-svg-icons/faCheck'

export interface SelectOption {
  value: string
  /** What the option reads as; defaults to `value`. */
  label?: string
  /** Muted text at the end of the row — a gloss, a reading, a count. */
  note?: string
  disabled?: boolean
}

export interface SelectProps {
  value: string
  /** Bare strings are treated as `{ value }`. */
  options: readonly (string | SelectOption)[]
  onChange: (value: string) => void
  /** Rendered beside the trigger, and labels it. */
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
 * A single-choice dropdown over Base UI's Select. Styling is the shared `.btn`
 * / `.popup` / `.option` component classes, so a caller's utility classes still
 * win over them.
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
  const items = options.map<SelectOption>((o) =>
    typeof o === 'string' ? { value: o } : o,
  )

  return (
    <BaseSelect.Root
      items={items.map((o) => ({ value: o.value, label: o.label ?? o.value }))}
      value={value}
      // Base UI allows clearing to null; this control is always populated.
      onValueChange={(next) => next !== null && onChange(next)}
      disabled={disabled}
      name={name}
    >
      <div className={clsx('flex items-center gap-2', className)}>
        {label !== undefined && (
          <BaseSelect.Label className="text-sm text-gray-500">
            {label}
          </BaseSelect.Label>
        )}
        <BaseSelect.Trigger
          className={clsx(
            'btn inline-flex items-center justify-between gap-2',
            triggerClassName,
          )}
        >
          <BaseSelect.Value placeholder={placeholder} />
          <BaseSelect.Icon>
            <Icon icon={faChevronDown} width="0.75em" height="0.75em" />
          </BaseSelect.Icon>
        </BaseSelect.Trigger>
      </div>

      <BaseSelect.Portal>
        <BaseSelect.Positioner className="z-20 outline-none" sideOffset={4}>
          <BaseSelect.Popup className="popup">
            <BaseSelect.List className="listbox">
              {items.map((o) => (
                <BaseSelect.Item
                  key={o.value}
                  value={o.value}
                  disabled={o.disabled}
                  className="option"
                >
                  <BaseSelect.ItemText className="min-w-6">
                    {o.label ?? o.value}
                  </BaseSelect.ItemText>
                  <BaseSelect.ItemIndicator>
                    <Icon icon={faCheck} width="0.75em" height="0.75em" />
                  </BaseSelect.ItemIndicator>
                  {o.note !== undefined && (
                    <span className="option__note">{o.note}</span>
                  )}
                </BaseSelect.Item>
              ))}
            </BaseSelect.List>
          </BaseSelect.Popup>
        </BaseSelect.Positioner>
      </BaseSelect.Portal>
    </BaseSelect.Root>
  )
}
