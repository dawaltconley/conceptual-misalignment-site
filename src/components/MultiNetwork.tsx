import type { Dictionary } from '@build/cedict'
import type { MasterSource } from '@lib/terms'
import type { NetworkData } from '@lib/networkx'
import Network, { NetworkSkeleton, Wrapper as NetworkWrapper } from './Network'
import useData from '@lib/browser/hooks/useData'
import { NetworkDataSchema } from '@lib/networkx'
import { useState, useMemo, type ReactNode } from 'react'
import useScrollPosition from '@lib/browser/hooks/useScrollPosition'
import { Tabs } from '@base-ui/react'
import SourceCard from './SourceCard'
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
  sources: MasterSource[]
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
  const sourceMap = useMemo(
    () => new Map(sources.map((s) => [s.id, s])),
    [sources],
  )
  const [selected, setSelected] = useState(sources[0].id)
  const active = sourceMap.get(selected)
  const { status, data, errorMessage } = useData(
    active?.data ?? '',
    assertNetworkData,
  )
  const network = data?.network ?? null

  return (
    <Tabs.Root
      value={selected}
      onValueChange={(v) => setSelected(v)}
      orientation="vertical"
      render={
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
            <Tabs.List
              render={
                <SourceSidebar align={sourceAlign}>
                  {sources.map((s) => (
                    <Tabs.Tab key={s.id} value={s.id}>
                      <SourceCard
                        source={s}
                        isHighlighted={s.id === selected}
                      />
                    </Tabs.Tab>
                  ))}
                </SourceSidebar>
              }
            />
          )}
        </Wrapper>
      }
    />
  )
}

function Wrapper({ children }: { children: ReactNode }): JSX.Element {
  return (
    <div className="items-top flex-row justify-between xl:flex xl:aspect-video">
      {children}
    </div>
  )
}

interface SourceSidebarProps {
  align: Side
  children: ReactNode
}

function SourceSidebar({ align, children }: SourceSidebarProps): JSX.Element {
  const { ref, position } = useScrollPosition<HTMLDivElement>({ threshold: 8 })
  return (
    <div
      ref={ref}
      className={clsx(
        'fade-mask mt-4 flex max-h-96 shrink-0 flex-row flex-wrap items-stretch justify-center gap-2 overflow-y-scroll p-1 pr-4 xl:mt-0 xl:h-full xl:max-h-none xl:basis-auto xl:flex-col xl:flex-nowrap xl:justify-normal',
        align === 'right'
          ? 'items-start xl:ml-2'
          : '-order-1 items-end xl:mr-2',
        {
          'fade-bottom-8': position === 'top',
          'fade-y-8': position === 'middle',
          'fade-top-8': position === 'bottom',
        },
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
