import type { NodeId } from '~/types/networkx'
import type { DictionaryEntry } from '@build/cedict'
import { useRef } from 'react'
import NetworkTooltip, { useNetworkTooltip } from './NetworkTooltip'
import HanziDefinition from './HanziDefinition'

interface HanziNodeProps {
  id: NodeId
  entry: DictionaryEntry
  maxEntries?: number
  isCentral?: boolean
}

export default function HanziNode({
  id,
  entry,
  isCentral = false,
  maxEntries = 7,
}: HanziNodeProps) {
  const nodeRef = useRef<HTMLSpanElement>(null)
  const { open, close, triggerProps } = useNetworkTooltip(id)
  const pinyin = entry.readings[0].pinyin

  return (
    <span
      ref={nodeRef}
      className="flex flex-col items-center py-1 leading-none"
      {...triggerProps}
    >
      <span>{id.toString()}</span>
      <span
        className={`mt-0.5 text-xs ${isCentral ? 'text-red-100' : 'text-gray-400'}`}
      >
        {pinyin}
      </span>
      <NetworkTooltip open={open} onDismiss={close} anchor={nodeRef}>
        <HanziDefinition entry={entry} maxDefinitions={maxEntries} />
      </NetworkTooltip>
    </span>
  )
}
