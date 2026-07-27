import type { Term } from '@lib/networkx'
import type { Dictionary } from '@build/cedict'
import Network, { NetworkSkeleton, Wrapper as NetworkWrapper } from './Network'
import useData from '@lib/browser/hooks/useData'
import { TermSchema } from '@lib/networkx'
import { useState, type ReactNode } from 'react'
import Button from './Button'
import clsx from 'clsx'

interface MultiNetworkProps {
  data: string
  sourceAlign?: 'left' | 'right'
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
        <NetworkWrapper>
          <div className="absolute inset-0 flex items-center justify-center">
            Error: {errorMessage}
          </div>
        </NetworkWrapper>
      </Wrapper>
    )

  return (
    <Wrapper>
      {status === 'loading' ? (
        <NetworkSkeleton />
      ) : (
        <Network
          centralNodeId={data.term}
          data={data.sources[selected].cooccurrence}
          dictionary={dictionary}
        />
      )}

      <div
        className={clsx(
          'mt-4 flex shrink flex-row flex-wrap gap-2 xl:mt-0 xl:flex-col',
          sourceAlign === 'right'
            ? 'items-start xl:ml-2'
            : '-order-1 items-end xl:mr-2',
        )}
      >
        {status === 'loading'
          ? new Array(8).fill(null).map((_, i) => <SourceSkeleton key={i} />)
          : data.sources.map((s, i) => (
              <Source
                key={s.url}
                title={s.title}
                description={s.description}
                url={new URL(s.url)}
                isActive={i === selected}
                onClick={() => setSelected(i)}
              />
            ))}
      </div>
    </Wrapper>
  )
}

function Wrapper({ children }: { children: ReactNode }): JSX.Element {
  return <div className="flex-row xl:flex">{children}</div>
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
