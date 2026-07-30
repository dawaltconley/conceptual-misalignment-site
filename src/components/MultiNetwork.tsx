import type { Dictionary } from '@build/cedict'
import type { NetworkData } from '@lib/networkx'
import Network, { NetworkSkeleton, Wrapper as NetworkWrapper } from './Network'
import useData from '@lib/browser/hooks/useData'
import { NetworkDataSchema } from '@lib/networkx'
import { useState, type ReactNode } from 'react'
import Button from './Button'
import clsx from 'clsx'

type Side = 'left' | 'right'

/** One selectable network: a label plus the path to its NetworkData JSON. */
export interface NetworkRef {
  id: string
  title: string
  /** Web path to a NetworkData JSON (co-occurrence or similarity — agnostic). */
  path: string
}

interface MultiNetworkProps {
  /** The networks to switch between. A single entry hides the sidebar. */
  sources: NetworkRef[]
  /** The node held at the centre of the graph (the term / rendering label). */
  centralNodeId: string
  sourceAlign?: Side
  dictionary?: Dictionary
}

/**
 * Displays a NetworkData graph, agnostic to its kind (co-occurrence or
 * similarity). Each source's graph lives in its own file, loaded on demand when
 * selected; with more than one source a sidebar switches between them (a single
 * source shows just the graph). Reset `selected` by remounting (a `key`).
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
    active?.path ?? '',
    assertNetworkData,
  )
  const network = data?.network ?? null

  return (
    <Wrapper>
      {sources.length === 0 ? (
        <NetworkError message={`No network for ${centralNodeId}`} />
      ) : status === 'loading' ? (
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

      {sources.length > 1 && (
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
      )}
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
