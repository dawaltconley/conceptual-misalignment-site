import type { Master } from '@lib/terms'
import type { Dictionary } from '@build/cedict'
import { useState } from 'react'
import MultiNetwork from './MultiNetwork'

interface TermNetworkProps {
  /** The master term index (passed from Astro, not fetched). */
  data: Master
  /** Dictionary for the Chinese hanzi nodes (pinyin + definitions). */
  dictionary?: Dictionary
}

/**
 * Side-by-side co-occurrence networks: a Chinese term and one of its English
 * renderings. Two dropdowns pick the term (the English options repopulate when
 * the Chinese term changes); each graph switches its own source via MultiNetwork.
 */
export default function TermNetwork({
  data,
  dictionary,
}: TermNetworkProps): JSX.Element {
  const terms = data.terms
  const [hanzi, setHanzi] = useState(terms[0]?.hanzi ?? '')
  const term = terms.find((t) => t.hanzi === hanzi) ?? terms[0]

  const [renderingLabel, setRenderingLabel] = useState(
    term?.renderings[0] ?? '',
  )
  const english =
    term?.english.find((e) => e.label === renderingLabel) ?? term?.english[0]

  function selectHanzi(next: string): void {
    setHanzi(next)
    const t = terms.find((t) => t.hanzi === next)
    if (t) setRenderingLabel(t.renderings[0]) // repopulate the English options
  }

  if (!term || !english) return <div>No term data.</div>

  return (
    <div className="items-start justify-center xl:flex xl:gap-6">
      <div className="xl:w-1/2">
        <TermSelect
          label="Chinese term"
          value={term.hanzi}
          options={terms.map((t) => t.hanzi)}
          onChange={selectHanzi}
        />
        <MultiNetwork
          key={term.hanzi}
          sources={term.chinese.sources}
          centralNodeId={term.hanzi}
          dictionary={dictionary}
          sourceAlign="left"
        />
      </div>
      <div className="mt-8 xl:mt-0 xl:w-1/2">
        <TermSelect
          label="English rendering"
          value={english.label}
          options={term.renderings}
          onChange={setRenderingLabel}
        />
        <MultiNetwork
          key={`${term.hanzi}:${english.label}`}
          sources={english.sources}
          centralNodeId={english.label}
          sourceAlign="right"
        />
      </div>
    </div>
  )
}

interface TermSelectProps {
  label: string
  value: string
  options: string[]
  onChange: (value: string) => void
}

function TermSelect({
  label,
  value,
  options,
  onChange,
}: TermSelectProps): JSX.Element {
  return (
    <label className="mb-3 flex items-center gap-2">
      <span className="text-sm text-gray-500">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded border border-gray-300 bg-white px-3 py-1.5 text-lg"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  )
}
