import { useId } from 'react'
import type { ReactNode } from 'react'
import { Combobox as BaseCombobox } from '@base-ui/react/combobox'
import clsx from 'clsx'
import Icon from './Icon'
import { faXmark } from '@fortawesome/pro-regular-svg-icons/faXmark'
import {
  ComboboxPopup,
  DEFAULT_LIMIT,
  optionLabel,
  useOptionFilter,
  type ComboboxOption,
} from './Combobox'

export interface TagsComboboxProps {
  /** Selected values, in the order they were added. */
  value: readonly string[]
  options: readonly ComboboxOption[]
  onChange: (values: string[]) => void
  /** Rendered beside the control, and labels its input. */
  label?: ReactNode
  placeholder?: string
  emptyMessage?: ReactNode
  /** Maximum matches rendered at once. */
  limit?: number
  disabled?: boolean
  name?: string
  /** Classes for the label + control stack. */
  className?: string
  /** Classes for the bordered control. */
  controlClassName?: string
}

/**
 * A multi-select combobox: chosen values sit in the control as removable tags.
 *
 * This is Base UI's `multiple` mode rather than anything hand-built — the
 * chips, their keyboard navigation, and Backspace-removes-the-last-tag all
 * come from `Chips`/`Chip`/`ChipRemove`.
 */
export default function TagsCombobox({
  value,
  options,
  onChange,
  label,
  placeholder,
  emptyMessage = 'No matches.',
  limit = DEFAULT_LIMIT,
  disabled,
  name,
  className,
  controlClassName,
}: TagsComboboxProps): JSX.Element {
  const inputId = useId()
  const filter = useOptionFilter()

  const byValue = new Map(options.map((o) => [o.value, o]))
  const selected = value
    .map((v) => byValue.get(v))
    .filter((o): o is ComboboxOption => o !== undefined)

  return (
    <BaseCombobox.Root
      multiple
      items={options as ComboboxOption[]}
      value={selected}
      onValueChange={(next) => onChange(next.map((o) => o.value))}
      itemToStringLabel={optionLabel}
      itemToStringValue={(option) => option.value}
      isItemEqualToValue={(a, b) => a.value === b.value}
      filter={filter}
      limit={limit}
      disabled={disabled}
      name={name}
    >
      <div className={clsx('flex flex-col gap-1', className)}>
        {label !== undefined && (
          <label htmlFor={inputId} className="text-sm text-gray-500">
            {label}
          </label>
        )}
        <BaseCombobox.InputGroup
          className={clsx('input-group', controlClassName)}
        >
          <BaseCombobox.Chips className="contents">
            <BaseCombobox.Value>
              {(items: ComboboxOption[]) => (
                <>
                  {items.map((option) => (
                    <BaseCombobox.Chip
                      key={option.value}
                      className="chip"
                      aria-label={optionLabel(option)}
                    >
                      {optionLabel(option)}
                      <BaseCombobox.ChipRemove
                        className="chip__remove"
                        aria-label={`Remove ${optionLabel(option)}`}
                      >
                        <Icon icon={faXmark} width="0.75em" height="0.75em" />
                      </BaseCombobox.ChipRemove>
                    </BaseCombobox.Chip>
                  ))}
                  <BaseCombobox.Input
                    id={inputId}
                    placeholder={placeholder}
                    className="input-group__input"
                  />
                </>
              )}
            </BaseCombobox.Value>
          </BaseCombobox.Chips>
        </BaseCombobox.InputGroup>
      </div>

      <ComboboxPopup emptyMessage={emptyMessage} />
    </BaseCombobox.Root>
  )
}
