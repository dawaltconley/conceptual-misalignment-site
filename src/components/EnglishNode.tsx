import type { NodeId } from '~/types/networkx'
import { useRef } from 'react'
import NetworkTooltip, { useNetworkTooltip } from './NetworkTooltip'

interface EnglishNodeProps {
  id: NodeId
  isCentral?: boolean
  variants?: string[]
}

export default function EnglishNode({
  id,
  variants,
}: EnglishNodeProps): JSX.Element {
  const term = id.toString()
  const nodeRef = useRef<HTMLDivElement>(null)
  const { open, close, triggerProps } = useNetworkTooltip(term)
  const hasVariants = !!variants?.length

  return (
    <span
      ref={nodeRef}
      {...triggerProps}
      className="flex flex-col items-center py-1 text-base leading-none"
    >
      {term}
      {hasVariants && '*'}
      {hasVariants && variants && (
        <NetworkTooltip
          open={open}
          onDismiss={close}
          anchor={nodeRef}
          sideOffset={16}
        >
          <div className="popup block border-gray-200 p-2.5 text-xs shadow-xl">
            Variants:
            <ul className="mt-1 list-inside list-disc">
              <li key={term}>{term}</li>
              {variants.map((v) => (
                <li key={v}>{v}</li>
              ))}
            </ul>
          </div>
        </NetworkTooltip>
      )}
    </span>
  )
}
