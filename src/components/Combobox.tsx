import { useId, useMemo, useState, type ReactNode } from 'react'
import { defaultFilter } from 'cmdk'
import { Check, ChevronsUpDown } from 'lucide-react'
import { cn } from '@lib/cn'
import { Button } from './ui/button'
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover'
import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandItem,
  CommandList,
} from './ui/command'

export interface ComboboxOption {
  value: string
  /** What the option reads as; defaults to `value`. */
  label?: ReactNode
  /**
   * Extra strings the option matches on, e.g. `pinyinKeywords('rén')` so a
   * hanzi option is reachable from a plain alphanumeric keyboard.
   */
  keywords?: string[]
  /** Muted text after the label — a gloss, a reading, a count. */
  description?: ReactNode
  disabled?: boolean
}

/** How many matches to render before the list is truncated. */
export const DEFAULT_MAX_RESULTS = 100

/**
 * Options that match `search`, best first, capped at `maxResults`.
 *
 * Filtering is done here rather than by cmdk's own pass so the number of
 * rendered items stays bounded: these lists run to the whole corpus
 * vocabulary, and cmdk keeps every item mounted. The scorer is still cmdk's
 * `defaultFilter`, which reads `keywords` alongside the value.
 */
export function filterOptions(
  options: readonly ComboboxOption[],
  search: string,
  maxResults = DEFAULT_MAX_RESULTS,
): ComboboxOption[] {
  if (!search.trim()) return options.slice(0, maxResults)
  return options
    .map((option) => ({
      option,
      score: defaultFilter(option.value, search, option.keywords),
    }))
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, maxResults)
    .map(({ option }) => option)
}

export interface ComboboxProps {
  /** Selected value, or `null` for none. */
  value: string | null
  options: readonly ComboboxOption[]
  onChange: (value: string) => void
  /** Rendered beside the trigger and wired up as its accessible name. */
  label?: ReactNode
  /** Trigger text when nothing is selected. */
  placeholder?: string
  searchPlaceholder?: string
  emptyMessage?: ReactNode
  maxResults?: number
  disabled?: boolean
  /** Classes for the label + trigger row. */
  className?: string
  /** Classes for the trigger itself. */
  triggerClassName?: string
  /** Classes for the popover panel. */
  contentClassName?: string
}

/**
 * A single-choice combobox: a Popover holding a `ui/command` list, filtered as
 * you type. This is shadcn's Combobox pattern — there is no combobox
 * primitive in Radix or single registry item to install — with the filtering
 * swapped for `filterOptions` so long lists stay bounded.
 */
export default function Combobox({
  value,
  options,
  onChange,
  label,
  placeholder = 'Select…',
  searchPlaceholder = 'Search…',
  emptyMessage = 'No matches.',
  maxResults,
  disabled,
  className,
  triggerClassName,
  contentClassName,
}: ComboboxProps): JSX.Element {
  const labelId = useId()
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')

  const matches = useMemo(
    () => filterOptions(options, search, maxResults),
    [options, search, maxResults],
  )
  const selected = options.find((o) => o.value === value)

  function select(next: string): void {
    onChange(next)
    setOpen(false)
    setSearch('')
  }

  return (
    <div className={cn('flex items-center gap-2', className)}>
      {label !== undefined && (
        <span id={labelId} className="text-sm text-muted-foreground">
          {label}
        </span>
      )}
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            aria-labelledby={label === undefined ? undefined : labelId}
            disabled={disabled}
            className={cn('justify-between font-normal', triggerClassName)}
          >
            <span className={cn('truncate', !selected && 'opacity-60')}>
              {selected ? (selected.label ?? selected.value) : placeholder}
            </span>
            <ChevronsUpDown className="opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className={cn('w-64 p-0', contentClassName)}>
          {/* cmdk's own filter is off; `matches` is already filtered and ranked. */}
          <Command shouldFilter={false}>
            <CommandInput
              value={search}
              onValueChange={setSearch}
              placeholder={searchPlaceholder}
            />
            <CommandList>
              <CommandEmpty>{emptyMessage}</CommandEmpty>
              {matches.map((option) => (
                <CommandItem
                  key={option.value}
                  value={option.value}
                  disabled={option.disabled}
                  onSelect={() => select(option.value)}
                >
                  <Check
                    className={cn(
                      option.value === value ? 'opacity-100' : 'opacity-0',
                    )}
                  />
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
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  )
}
