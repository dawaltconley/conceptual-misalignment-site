import type { NodeId } from '~/types/networkx'
import type { DictionaryEntry } from '@build/cedict'
import { useState, useRef } from 'react'
import { createPortal } from 'react-dom'
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
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [tooltipRect, setTooltipRect] = useState<DOMRect | null>(null)

  function handlePointerEnter() {
    timerRef.current = setTimeout(() => {
      setTooltipRect(nodeRef.current?.getBoundingClientRect() ?? null)
    }, 1000)
  }

  function handlePointerLeave() {
    if (timerRef.current) clearTimeout(timerRef.current)
    setTooltipRect(null)
  }

  return (
    <span
      ref={nodeRef}
      className="flex flex-col items-center py-1 leading-none"
      onPointerEnter={handlePointerEnter}
      onPointerLeave={handlePointerLeave}
      onPointerDown={handlePointerLeave}
    >
      <span>{id.toString()}</span>
      <span
        className={`mt-0.5 text-xs ${isCentral ? 'text-red-100' : 'text-gray-400'}`}
      >
        {entry.pinyin}
      </span>
      {tooltipRect &&
        createPortal(
          <div
            className="pointer-events-none fixed z-50"
            style={{
              top: tooltipRect.top - 8,
              left: tooltipRect.left + tooltipRect.width / 2,
              transform: 'translate(-50%, -100%)',
            }}
          >
            <HanziDefinition entry={entry} maxDefinitions={maxEntries} />
          </div>,
          document.body,
        )}
    </span>
  )
}
