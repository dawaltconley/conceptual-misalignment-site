import { useState, useEffect, useRef, useMemo, type ReactNode } from 'react'
import { TSNE } from '@keckelt/tsne'
import clsx from 'clsx'
import useData from '@lib/browser/hooks/useData'
import { EmbeddingDatasetSchema } from '@lib/embeddings'
import {
  ScatterSkeleton,
  Wrapper as ScatterWrapper,
  type ScatterPoint,
} from './ScatterPlot'
import ScatterPlot from './CanvasScatterPlot'
import ScatterLegend, { type LegendLabel } from './ScatterLegend'
import * as d3 from 'd3'

type Layout = 'pca' | 'tsne'

const TSNE_MAX_ITER = 500
const TSNE_STEPS_PER_FRAME = 2

interface EmbeddingScatterProps {
  data: string
}

/**
 * Data-loading + controls wrapper (the `MultiNetwork.tsx` of scatter plots).
 * Fetches one reduced-vector dataset and derives the 2-D layout it hands to the
 * pure `ScatterPlot`: PCA is free (columns 0/1 of the variance-ordered vector);
 * t-SNE is run client-side over the full vector with a tunable perplexity.
 */
export default function EmbeddingScatter({
  data: dataPath,
}: EmbeddingScatterProps): JSX.Element {
  const { status, data, errorMessage } = useData(dataPath, (d) =>
    EmbeddingDatasetSchema.parse(d),
  )
  const [layout, setLayout] = useState<Layout>('pca')
  const [perplexity, setPerplexity] = useState(30)
  const [tsnePoints, setTsnePoints] = useState<ScatterPoint[] | null>(null)
  const [iter, setIter] = useState(0)
  const [highlighted, setHighlighted] = useState<Set<number>>(new Set())

  // PCA layout — free: the reduced columns are variance-ordered, so [0],[1]
  // are the 2-D principal-component coordinates.
  const pcaPoints = useMemo<ScatterPoint[]>(
    () =>
      data
        ? data.nodes.map((n) => ({
            id: n.id,
            community: n.community,
            target: n.target,
            x: n.vec[0] ?? 0,
            y: n.vec[1] ?? 0,
          }))
        : [],
    [data],
  )

  const communityLegend = useMemo<LegendLabel[]>(() => {
    const communities: string[][] = []
    data?.nodes.forEach((n) => {
      communities[n.community] ??= []
      communities[n.community].push(n.id)
    })
    return communities.map<LegendLabel>((terms, community) => ({
      id: community.toString(),
      color: getColor(community),
      description: `C${community}: ${terms.sort().join(', ')}`,
    }))
  }, [data])

  // t-SNE layout — iterative; steps a few times per animation frame so the user
  // watches it converge. Re-initialises on perplexity change or a manual re-run.
  useEffect(() => {
    if (layout !== 'tsne') return
    runTSNE()
    return () => {
      cancelTSNE()
    }
  }, [data])

  const handleLayout = (layout: Layout): void => {
    if (layout === 'tsne') {
      runTSNE()
    } else {
      cancelTSNE()
    }
    setLayout(layout)
  }

  const raf = useRef(0)
  const runTSNE = (): void => {
    if (!data) return
    const nodes = data.nodes
    const tsne = new TSNE({ epsilon: 10, perplexity, dim: 2 })
    tsne.initDataRaw(nodes.map((n) => n.vec))

    let steps = 0
    let cancelled = false
    setIter(0)

    const tick = () => {
      if (cancelled) return
      for (let i = 0; i < TSNE_STEPS_PER_FRAME; i++) {
        tsne.step()
        steps++
      }
      const Y = tsne.getSolution() as number[][]
      setTsnePoints(
        nodes.map((n, i) => ({
          id: n.id,
          community: n.community,
          target: n.target,
          x: Y[i][0],
          y: Y[i][1],
        })),
      )
      setIter(steps)
      if (steps < TSNE_MAX_ITER) raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
  }
  const cancelTSNE = (): void => {
    cancelAnimationFrame(raf.current)
  }

  if (status === 'error') {
    return (
      <Wrapper>
        <ScatterWrapper>
          <div className="absolute inset-0 flex items-center justify-center">
            Error: {errorMessage}
          </div>
        </ScatterWrapper>
      </Wrapper>
    )
  }

  if (status === 'loading' || !data) {
    return (
      <Wrapper>
        <ScatterSkeleton />
      </Wrapper>
    )
  }

  const points = layout === 'pca' ? pcaPoints : (tsnePoints ?? [])
  const converging = layout === 'tsne' && iter < TSNE_MAX_ITER

  return (
    <Wrapper>
      <ScatterPlot
        points={points}
        getColor={getColor}
        highlightedCommunities={highlighted}
      />
      <div className="ml-8 columns-2 text-xs">
        <ScatterLegend
          labels={communityLegend}
          onHover={(id) =>
            setHighlighted(id ? new Set([Number(id)]) : new Set())
          }
        />
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <div className="flex gap-2">
          <Toggle
            label="PCA"
            active={layout === 'pca'}
            onClick={() => handleLayout('pca')}
          />
          <Toggle
            label="t-SNE"
            active={layout === 'tsne'}
            onClick={() => handleLayout('tsne')}
          />
        </div>

        {layout === 'tsne' && (
          <>
            <label className="flex items-center gap-2 text-sm">
              perplexity
              <input
                type="range"
                min={5}
                max={50}
                value={perplexity}
                onChange={(e) => setPerplexity(Number(e.target.value))}
              />
              <span className="w-6 tabular-nums">{perplexity}</span>
            </label>
            <button
              className="rounded border border-gray-900 px-2 py-0.5 text-sm hover:bg-red-200"
              onClick={runTSNE}
            >
              re-run
            </button>
            <span className="text-sm tabular-nums text-gray-500">
              {converging
                ? `iterating… ${iter}/${TSNE_MAX_ITER}`
                : `done (${iter})`}
            </span>
          </>
        )}
      </div>
    </Wrapper>
  )
}

function Wrapper({ children }: { children: ReactNode }): JSX.Element {
  return <div className="w-full">{children}</div>
}

interface ToggleProps {
  label: string
  active?: boolean
  onClick: () => void
}

function Toggle({ label, active = false, onClick }: ToggleProps): JSX.Element {
  return (
    <button
      className={clsx(
        'rounded border border-gray-900 px-2 py-0.5 text-sm duration-150 hover:bg-red-200',
        active && 'border-red-500 bg-red-500 text-white',
      )}
      onClick={onClick}
    >
      {label}
    </button>
  )
}

const color = d3.scaleOrdinal<number, string>(d3.schemeTableau10)
const NO_COMMUNITY_COLOR = '#cbd5e1' // grey for isolates (community -1)

function getColor(community: number): string {
  return community < 0 ? NO_COMMUNITY_COLOR : color(community)
}
