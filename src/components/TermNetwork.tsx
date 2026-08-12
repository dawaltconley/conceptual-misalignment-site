import type { Master, CorpusSide } from '@lib/terms'
import type { Range } from '@lib/graphs'
import type { Dictionary } from '@build/cedict'
import type { NetworkProps } from './Network'
import { useState } from 'react'
import MultiNetwork from './MultiNetwork'
import Select from './Select'
import { getHeading, type Heading } from '@lib/headings'

type NetworkKind = 'cooccurrence' | 'similarity'

interface TermNetworkProps extends Omit<
  NetworkProps,
  'data' | 'centralNodeId'
> {
  title: string
  /** The master term index (passed from Astro, not fetched). */
  data: Master
  /**
   * Which networks to show: per-source co-occurrence (a sidebar of sources) or
   * the single whole-corpus similarity network (no sidebar). Default co-occurrence.
   */
  kind?: NetworkKind
  /** Dictionary for the Chinese hanzi nodes (pinyin + definitions). */
  dictionary?: Dictionary

  headingLevel?: Heading
  actualEdgeWeightRange?: Range
  targetEdgeWeightRange?: Range
}

/** The networks to feed one MultiNetwork for a corpus side, per kind. */
function refsFor(side: CorpusSide, kind: NetworkKind) {
  return kind === 'similarity' ? side.similarity : side.cooccurrence
}

/**
 * Side-by-side networks for a Chinese term and one of its English renderings.
 * Two dropdowns pick the term (the English options repopulate when the Chinese
 * term changes); `kind` selects co-occurrence (per-source) or similarity (single).
 */
export default function TermNetwork({
  title,
  data,
  kind = 'cooccurrence',
  dictionary,
  headingLevel,
  actualEdgeWeightRange,
  targetEdgeWeightRange,
}: TermNetworkProps): JSX.Element {
  const H = headingLevel ? getHeading(headingLevel) : 'p'

  const terms = data.terms
  const [hanzi, setHanzi] = useState(terms[0]?.hanzi ?? '')
  const term = terms.find((t) => t.hanzi === hanzi) ?? terms[0]

  const [renderingLabel, setRenderingLabel] = useState(
    term?.renderings[0] ?? '',
  )
  const english =
    term?.english.find((e) => e.term.label === renderingLabel) ??
    term?.english[0]

  function selectHanzi(next: string): void {
    setHanzi(next)
    const t = terms.find((t) => t.hanzi === next)
    if (t) setRenderingLabel(t.renderings[0]) // repopulate the English options
  }

  if (!term || !english) return <div>No term data.</div>

  const chinesePercentage = (
    (english.chinesePhilosophyOccurrences / english.totalOccurrences) *
    100
  ).toFixed(2)

  return (
    <div className="overflow-visible rounded-md border border-gray-100 bg-white">
      <div className="grid 2xl:grid-cols-2">
        <div className="sticky top-0 z-50 grid grid-cols-2 flex-row items-end justify-between gap-4 border-b border-gray-100 bg-white p-4 lg:flex">
          <H className="mr-auto text-xl font-bold">{title}</H>
          <div className="order-last col-span-2 mx-auto flex shrink-0 flex-row gap-[inherit] lg:order-none lg:col-span-1">
            <Select
              label="Chinese"
              value={term.hanzi}
              options={terms.map((t) => t.hanzi)}
              onChange={selectHanzi}
              className="shrink-0"
              triggerClassName="text-lg"
            />
            <Select
              label="English"
              value={english.term.label}
              options={term.renderings}
              onChange={setRenderingLabel}
              className="shrink-0"
              triggerClassName="text-lg"
            />
          </div>
          <div className="ml-auto max-w-80 text-right text-sm leading-4 text-gray-700">
            <span className="align-baseline text-lg font-bold leading-4 text-gray-900">
              {chinesePercentage}%
            </span>{' '}
            sampled usage of{' '}
            <span className="italic">“{english.term.label}”</span> comes from
            articles about Chinese philosophy
          </div>
        </div>
        <div className="p2">
          <MultiNetwork
            key={`${kind}:${term.hanzi}`}
            sources={refsFor(term.chinese, kind)}
            centralNodeId={term.hanzi}
            dictionary={dictionary}
            actualEdgeWeightRange={actualEdgeWeightRange}
            targetEdgeWeightRange={targetEdgeWeightRange}
            sourceAlign="left"
          />
        </div>
        <div className="p-2">
          <MultiNetwork
            key={`${kind}:${term.hanzi}:${english.term.label}`}
            sources={refsFor(english, kind)}
            centralNodeId={english.term.label}
            actualEdgeWeightRange={actualEdgeWeightRange}
            targetEdgeWeightRange={targetEdgeWeightRange}
            sourceAlign="right"
          />
        </div>
      </div>
    </div>
  )
}
