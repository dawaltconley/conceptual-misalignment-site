import { useCallback, useId } from 'react'
import type { ReactNode } from 'react'
import { Combobox as BaseCombobox } from '@base-ui/react/combobox'
import clsx from 'clsx'
import Icon from './Icon'
import { faChevronDown } from '@fortawesome/pro-regular-svg-icons/faChevronDown'
import { faCheck } from '@fortawesome/pro-regular-svg-icons/faCheck'

export interface ComboboxOption {
  value: string
  /** What the option reads as; defaults to `value`. */
  label?: string
  /**
   * Extra strings the option matches on, e.g. `pinyinKeywords('rén')` so a
   * hanzi option is reachable from a plain alphanumeric keyboard.
   */
  keywords?: string[]
  /** Muted text at the end of the row — a gloss, a reading, a count. */
  note?: string
}

/** How many matches Base UI renders before truncating the list. */
export const DEFAULT_LIMIT = 100

export const optionLabel = (option: ComboboxOption): string =>
  option.label ?? option.value

/**
 * Matches an option against what's typed, using Base UI's collator-backed
 * `contains`. At `sensitivity: 'base'` that ignores case *and* diacritics, so
 * `ren` finds `rén` before pinyin keywords are even consulted.
 */
export function useOptionFilter(): (
  option: ComboboxOption,
  query: string,
) => boolean {
  const { contains } = BaseCombobox.useFilter({ sensitivity: 'base' })
  return useCallback(
    (option, query) =>
      contains(option.value, query) ||
      contains(optionLabel(option), query) ||
      (option.keywords ?? []).some((keyword) => contains(keyword, query)),
    [contains],
  )
}

/** The list body, shared with `TagsCombobox`. */
export function ComboboxPopup({
  emptyMessage,
  className,
}: {
  emptyMessage: ReactNode
  className?: string
}): JSX.Element {
  return (
    <BaseCombobox.Portal>
      <BaseCombobox.Positioner className="z-20 outline-none" sideOffset={4}>
        <BaseCombobox.Popup className={clsx('popup', className)}>
          <BaseCombobox.Empty className="px-3 py-4 text-sm text-gray-500 empty:p-0">
            {emptyMessage}
          </BaseCombobox.Empty>
          <BaseCombobox.List className="listbox">
            {(option: ComboboxOption) => (
              <BaseCombobox.Item
                key={option.value}
                value={option}
                className="option"
              >
                <span className="min-w-6">{optionLabel(option)}</span>
                {option.note !== undefined && (
                  <span className="option__note">{option.note}</span>
                )}
                <BaseCombobox.ItemIndicator>
                  <Icon icon={faCheck} width="0.75em" height="0.75em" />
                </BaseCombobox.ItemIndicator>
              </BaseCombobox.Item>
            )}
          </BaseCombobox.List>
        </BaseCombobox.Popup>
      </BaseCombobox.Positioner>
    </BaseCombobox.Portal>
  )
}

export interface ComboboxProps {
  /** Selected value, or `null` for none. */
  value: string | null
  options: readonly ComboboxOption[]
  onChange: (value: string | null) => void
  /** Rendered beside the control, and labels its input. */
  label?: ReactNode
  placeholder?: string
  emptyMessage?: ReactNode
  /** Maximum matches rendered at once. */
  limit?: number
  disabled?: boolean
  name?: string
  /** Classes for the label + control row. */
  className?: string
  /** Classes for the bordered control. */
  controlClassName?: string
}

/**
 * A single-choice combobox over Base UI's Combobox: an input that filters the
 * list as you type. Values are plain strings on the outside; the option
 * objects are what Base UI sees, so `note` and `keywords` ride along.
 */
export default function Combobox({
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
}: ComboboxProps): JSX.Element {
  const inputId = useId()
  const filter = useOptionFilter()
  const selected = options.find((o) => o.value === value) ?? null

  return (
    <BaseCombobox.Root
      items={options as ComboboxOption[]}
      value={selected}
      onValueChange={(option) => onChange(option?.value ?? null)}
      itemToStringLabel={optionLabel}
      itemToStringValue={(option) => option.value}
      isItemEqualToValue={(a, b) => a.value === b.value}
      filter={filter}
      limit={limit}
      disabled={disabled}
      name={name}
    >
      <div className={clsx('flex items-center gap-2', className)}>
        {label !== undefined && (
          <label htmlFor={inputId} className="text-sm text-gray-500">
            {label}
          </label>
        )}
        <BaseCombobox.InputGroup
          className={clsx('input-group', controlClassName)}
        >
          <BaseCombobox.Input
            id={inputId}
            placeholder={placeholder}
            className="input-group__input"
          />
          <BaseCombobox.Trigger
            aria-label="Open list"
            className="input-group__button"
          >
            <Icon icon={faChevronDown} width="0.75em" height="0.75em" />
          </BaseCombobox.Trigger>
        </BaseCombobox.InputGroup>
      </div>

      <ComboboxPopup emptyMessage={emptyMessage} />
    </BaseCombobox.Root>
  )
}
