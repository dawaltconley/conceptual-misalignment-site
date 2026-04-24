import { useState, useRef } from 'react'
import { createPortal } from 'react-dom'
import type { NodeId } from '~/types/networkx'
import type { DictionaryEntry } from '@build/cedict'

interface HanziNodeProps {
  id: NodeId
  entry: DictionaryEntry
  isCentral?: boolean
}

export default function HanziNode({ id, entry, isCentral }: HanziNodeProps) {
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
      className="flex flex-col items-center leading-none"
      onPointerEnter={handlePointerEnter}
      onPointerLeave={handlePointerLeave}
      onPointerDown={handlePointerLeave}
    >
      <span>{String(id)}</span>
      <span className={`mt-0.5 text-[0.6rem] ${isCentral ? 'text-red-100' : 'text-gray-400'}`}>
        {entry.pinyin}
      </span>
      {tooltipRect &&
        createPortal(
          <div
            className="pointer-events-none fixed z-50 w-52 rounded-lg bg-white p-2.5 text-xs shadow-xl ring-1 ring-gray-200"
            style={{
              top: tooltipRect.top - 8,
              left: tooltipRect.left + tooltipRect.width / 2,
              transform: 'translate(-50%, -100%)',
            }}
          >
            <p className="mb-1 font-semibold text-gray-900">{entry.pinyin}</p>
            <p className="text-gray-500">{entry.definitions.slice(0, 3).join('; ')}</p>
          </div>,
          document.body,
        )}
    </span>
  )
}
