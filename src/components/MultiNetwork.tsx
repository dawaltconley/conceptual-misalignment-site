import type { Dictionary } from '@build/cedict'
import type { MasterSource } from '@lib/terms'
import type { NetworkData } from '@lib/networkx'
import Network, { NetworkSkeleton, Wrapper as NetworkWrapper } from './Network'
import useData from '@lib/browser/hooks/useData'
import { NetworkDataSchema } from '@lib/networkx'
import { useState, type ReactNode } from 'react'
import Button from './Button'
import clsx from 'clsx'

type Side = 'left' | 'right'

interface MultiNetworkProps {
  /** The term's sources (from the master index); the sidebar switches between them. */
  sources: MasterSource[]
  /** The node held at the centre of the graph (the term / rendering label). */
  centralNodeId: string
  sourceAlign?: Side
  dictionary?: Dictionary
}

/**
 * A co-occurrence network with a sidebar to switch between the term's sources.
 * Each source's graph lives in its own file (`MasterSource.cooccurrence`), loaded
 * on demand when selected. Reset `selected` by remounting (a `key` on the term).
 */
export default function MultiNetwork({
  sources,
  centralNodeId,
  sourceAlign = 'right',
  dictionary,
}: MultiNetworkProps): JSX.Element {
  const [selected, setSelected] = useState(0)
  const active = sources[selected]
  const { status, data, errorMessage } = useData(
    active?.cooccurrence ?? '',
    assertNetworkData,
  )
  const network = data?.network ?? null

  return (
    <Wrapper>
      {status === 'loading' ? (
        <NetworkSkeleton />
      ) : status === 'error' ? (
        <NetworkError message={errorMessage} />
      ) : network ? (
        <Network
          centralNodeId={centralNodeId}
          data={network}
          dictionary={dictionary}
        />
      ) : (
        <NetworkError message={`No occurrences of ${centralNodeId}`} />
      )}

      <SourceSidebar align={sourceAlign}>
        {sources.map((s, i) => (
          <Source
            key={s.id}
            title={s.title}
            isActive={i === selected}
            onClick={() => setSelected(i)}
          />
        ))}
      </SourceSidebar>
    </Wrapper>
  )
}

function Wrapper({ children }: { children: ReactNode }): JSX.Element {
  return <div className="flex-row xl:flex">{children}</div>
}

interface SourceSidebarProps {
  align: Side
  children: ReactNode
}

function SourceSidebar({ align, children }: SourceSidebarProps): JSX.Element {
  return (
    <div
      className={clsx(
        'mt-4 flex shrink flex-row flex-wrap gap-2 xl:mt-0 xl:flex-col',
        align === 'right'
          ? 'items-start xl:ml-2'
          : '-order-1 items-end xl:mr-2',
      )}
    >
      {children}
    </div>
  )
}

interface NetworkErrorProps {
  message?: string
}

function NetworkError({ message }: NetworkErrorProps): JSX.Element {
  return (
    <NetworkWrapper>
      <div className="absolute inset-0 flex items-center justify-center">
        {message ?? 'Error'}
      </div>
    </NetworkWrapper>
  )
}

function assertNetworkData(data: unknown): NetworkData {
  return NetworkDataSchema.parse(data)
}

interface SourceProps {
  title: string
  isActive?: boolean
  onClick: () => void
}

function Source({ title, isActive, onClick }: SourceProps): JSX.Element {
  return (
    <Button
      className="max-w-40 p-1 xl:block"
      isActive={isActive}
      onClick={onClick}
    >
      {title}
    </Button>
  )
}
