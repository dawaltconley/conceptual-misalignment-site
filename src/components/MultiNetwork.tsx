import type {Term} from '@lib/networkx'
import type {Dictionary} from '@build/cedict'
import Network from './Network'
import {useState} from 'react'
import clsx from 'clsx'

interface MultiNetworkProps {
  data: Term
  sourceAlign?: 'left' | 'right'
  dictionary?: Dictionary
}

export default function MultiNetwork({
  data,
  sourceAlign = 'right',
  dictionary,
}: MultiNetworkProps): JSX.Element {
  const [selected, setSelected] = useState(0)
  const network = data.sources[selected].co_occurance

  return (
    <div className="flex-row xl:flex">
      <Network
        centralNodeId={data.term}
        data={network}
        dictionary={dictionary}
      />

      <div
        className={clsx(
          'mt-4 flex shrink flex-row flex-wrap gap-2 xl:mt-0 xl:flex-col',
          sourceAlign === 'right' ? 'items-start' : '-order-1 items-end',
        )}
      >
        {data.sources.map((s, i) => (
          <MultiNetworkSource
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

interface MultiNetworkSourceProps {
  url: URL
  title: string
  description: string
  isActive?: boolean
  onClick: () => void
}

function MultiNetworkSource({
  title,
  isActive = false,
  onClick,
}: MultiNetworkSourceProps): JSX.Element {
  return (
    <button
      className={clsx(
        'max-w-40 overflow-hidden overflow-ellipsis whitespace-nowrap rounded border border-gray-900 p-1 duration-150 hover:bg-red-200 xl:block',
        isActive && 'border-red-500 bg-red-500 text-white',
      )}
      onClick={onClick}
    >
      {title}
    </button>
  )
}
