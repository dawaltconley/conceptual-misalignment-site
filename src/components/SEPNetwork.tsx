import type { Term } from '@lib/networkx'
import type { Dictionary } from '@build/cedict'
import Network from './Network'
import { useState } from 'react'
import clsx from 'clsx'

interface SEPNetworkProps {
  data: Term
  sourceAlign?: 'left' | 'right'
  dictionary?: Dictionary
}

export default function SEPNetwork({
  data,
  sourceAlign = 'right',
  dictionary,
}: SEPNetworkProps): JSX.Element {
  const [selected, setSelected] = useState(0)
  const network = data.sources[selected].co_occurance

  return (
    <div className="flex flex-row">
      <Network
        centralNodeId={data.term}
        data={network}
        dictionary={dictionary}
      />

      <div
        className={clsx(
          'shrink space-y-2',
          sourceAlign === 'left' && '-order-1',
        )}
      >
        {data.sources.map((s, i) => (
          <SEPPage
            key={s.url}
            title={s.title}
            description={s.description}
            url={new URL(s.url)}
            isActive={i === selected}
            onClick={() => setSelected(i)}
          />
        ))}
      </div>
    </div>
  )
}

interface SEPPageProps {
  url: URL
  title: string
  description: string
  isActive?: boolean
  onClick: () => void
}

function SEPPage({
  title,
  isActive = false,
  onClick,
}: SEPPageProps): JSX.Element {
  return (
    <button
      className={clsx(
        'block max-w-40 overflow-hidden overflow-ellipsis whitespace-nowrap rounded border border-gray-900 p-1 duration-150 hover:bg-red-200',
        isActive && 'border-red-500 bg-red-500 text-white',
      )}
      onClick={onClick}
    >
      {title}
    </button>
  )
}
