import type { Term } from '@lib/networkx'
import type { Dictionary } from '@build/cedict'
import Network, { NetworkSkeleton, Wrapper as NetworkWrapper } from './Network'
import useData from '@lib/browser/hooks/useData'
import { TermSchema } from '@lib/networkx'
import { useState, type ReactNode } from 'react'
import Button from './Button'
import clsx from 'clsx'

type Side = 'left' | 'right'

interface MultiNetworkProps {
  data: string
  sourceAlign?: Side
  dictionary?: Dictionary
}

export default function MultiNetwork({
  data: dataPath,
  sourceAlign = 'right',
  dictionary,
}: MultiNetworkProps): JSX.Element {
  const [selected, setSelected] = useState(0)
  const { status, data, errorMessage } = useData(dataPath, assertTerm)

  if (status === 'error')
    return (
      <Wrapper>
        <NetworkError message={errorMessage} />
      </Wrapper>
    )

  if (status === 'loading')
    return (
      <Wrapper>
        <NetworkSkeleton />
        <SourceSidebar align={sourceAlign}>
          {new Array(8).fill(null).map((_, i) => (
            <SourceSkeleton key={i} />
          ))}
        </SourceSidebar>
      </Wrapper>
    )

  const network = data.sources[selected].cooccurrence

  return (
    <Wrapper>
      {network ? (
        <Network
          centralNodeId={data.term}
          data={network}
          dictionary={dictionary}
        />
      ) : (
        <NetworkError message={`No occurances of ${data.term}`} />
      )}

      <SourceSidebar align={sourceAlign}>
        {data.sources.map((s, i) => (
          <Source
            key={s.url}
            title={s.title}
            description={s.description}
            url={new URL(s.url)}
            isActive={i === selected}
            onClick={() => setSelected(i)}
            disabled={!s.cooccurrence}
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
  message: string
}

function NetworkError({ message }: NetworkErrorProps): JSX.Element {
  return (
    <NetworkWrapper>
      <div className="absolute inset-0 flex items-center justify-center">
        Error: {message}
      </div>
    </NetworkWrapper>
  )
}

function assertTerm(data: any): Term {
  return TermSchema.parse(data)
}

interface SourceProps {
  url: URL
  title: string
  description: string
  isActive?: boolean
  onClick: () => void
  disabled?: boolean
}

function Source({ title, ...props }: SourceProps): JSX.Element {
  return (
    <Button className="max-w-40 p-1 xl:block" {...props}>
      {title}
    </Button>
  )
}

const SourceSkeleton = (): JSX.Element => (
  <div className="skeleton inline-block h-[34px] w-24 rounded xl:block" />
)
