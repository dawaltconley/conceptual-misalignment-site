import {
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from 'react'
import { Command as CommandPrimitive } from 'cmdk'
import { X } from 'lucide-react'
import { cn } from '@lib/cn'
import { filterOptions, type ComboboxOption } from './Combobox'
import { Badge } from './ui/badge'
import { Command, CommandEmpty, CommandItem, CommandList } from './ui/command'

export interface TagsComboboxProps {
  /** Selected values, in the order they were added. */
  value: readonly string[]
  options: readonly ComboboxOption[]
  onChange: (values: string[]) => void
  label?: ReactNode
  placeholder?: string
  emptyMessage?: ReactNode
  maxResults?: number
  /** Stop accepting new tags past this many. */
  maxTags?: number
  disabled?: boolean
  className?: string
}

/**
 * A multi-select combobox: chosen values sit in the control as removable tags,
 * and the rest are filtered in a listbox below as you type.
 *
 * The input lives *inside* the `Command` root rather than behind a popover
 * trigger, which is what lets one text field both hold the tags and drive
 * cmdk's list navigation. Selected options are dropped from the list instead
 * of being shown as checked — with tags visible in the control, a checked row
 * would say the same thing twice.
 */
export default function TagsCombobox({
  value,
  options,
  onChange,
  label,
  placeholder = 'Search…',
  emptyMessage = 'No matches.',
  maxResults,
  maxTags,
  disabled,
  className,
}: TagsComboboxProps): JSX.Element {
  const inputId = useId()
  const inputRef = useRef<HTMLInputElement>(null)
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')

  const selected = useMemo(() => new Set(value), [value])
  const full = maxTags !== undefined && value.length >= maxTags

  const matches = useMemo(
    () =>
      full
        ? []
        : filterOptions(
            options.filter((o) => !selected.has(o.value)),
            search,
            maxResults,
          ),
    [options, selected, search, maxResults, full],
  )

  const labelFor = (v: string): ReactNode =>
    options.find((o) => o.value === v)?.label ?? v

  function add(next: string): void {
    if (full || selected.has(next)) return
    onChange([...value, next])
    setSearch('')
  }

  function remove(target: string): void {
    onChange(value.filter((v) => v !== target))
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>): void {
    if (event.key === 'Escape') {
      setOpen(false)
      return
    }
    // Backspace only eats a tag once the text is gone, so it can't swallow a
    // character and a tag in the same keystroke.
    if (event.key === 'Backspace' && !search && value.length) {
      remove(value[value.length - 1])
    }
  }

  return (
    <div className={cn('flex flex-col gap-1', className)}>
      {label !== undefined && (
        <label htmlFor={inputId} className="text-sm text-muted-foreground">
          {label}
        </label>
      )}
      <Command
        shouldFilter={false}
        className="overflow-visible bg-transparent"
        // Enter would otherwise submit an enclosing form before cmdk selects.
        onKeyDown={(e) => {
          if (e.key === 'Enter') e.preventDefault()
        }}
      >
        <div
          className={cn(
            'flex flex-wrap items-center gap-1 rounded-md border border-input bg-transparent px-2 py-1.5 text-sm shadow-sm',
            'focus-within:ring-1 focus-within:ring-ring',
            disabled && 'cursor-not-allowed opacity-50',
          )}
          onClick={() => inputRef.current?.focus()}
        >
          {value.map((v) => (
            <Badge key={v} variant="secondary" className="gap-1 font-normal">
              {labelFor(v)}
              <button
                type="button"
                aria-label={`Remove ${v}`}
                disabled={disabled}
                className="rounded-sm opacity-60 hover:opacity-100 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                onClick={(e) => {
                  e.stopPropagation()
                  remove(v)
                }}
              >
                <X className="size-3" />
              </button>
            </Badge>
          ))}
          <CommandPrimitive.Input
            ref={inputRef}
            id={inputId}
            value={search}
            onValueChange={setSearch}
            onKeyDown={handleKeyDown}
            onFocus={() => setOpen(true)}
            onBlur={() => setOpen(false)}
            disabled={disabled || full}
            placeholder={full ? '' : placeholder}
            className="min-w-24 flex-1 bg-transparent outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed"
          />
        </div>

        <div className="relative">
          {open && !full && (
            <CommandList className="absolute top-1 z-50 w-full rounded-md border bg-popover text-popover-foreground shadow-md">
              <CommandEmpty>{emptyMessage}</CommandEmpty>
              {matches.map((option) => (
                <CommandItem
                  key={option.value}
                  value={option.value}
                  disabled={option.disabled}
                  // Blur fires before click, so the list would unmount first.
                  onMouseDown={(e) => e.preventDefault()}
                  onSelect={() => add(option.value)}
                >
                  <span className="truncate">
                    {option.label ?? option.value}
                  </span>
                  {option.description !== undefined && (
                    <span className="ml-auto truncate text-xs text-muted-foreground">
                      {option.description}
                    </span>
                  )}
                </CommandItem>
              ))}
            </CommandList>
          )}
        </div>
      </Command>
    </div>
  )
}
