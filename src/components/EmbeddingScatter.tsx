import type { Dictionary } from '@lib/build/cedict'
import { useState, useEffect, useMemo, type ReactNode } from 'react'
import useData from '@lib/browser/hooks/useData'
import useTsne from '@lib/browser/hooks/useTsne'
import { EmbeddingDatasetSchema } from '@lib/embeddings'
import { Button } from './ui/button'
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

interface EmbeddingScatterProps {
  data: string
  dictionary?: Dictionary
}

/**
 * Data-loading + controls wrapper (the `MultiNetwork.tsx` of scatter plots).
 * Fetches one reduced-vector dataset and derives the 2-D layout it hands to the
 * pure `ScatterPlot`: PCA is free (columns 0/1 of the variance-ordered vector);
 * t-SNE is run client-side over the full vector with a tunable perplexity.
 */
export default function EmbeddingScatter({
  data: dataPath,
  dictionary,
}: EmbeddingScatterProps): JSX.Element {
  const { status, data, errorMessage } = useData(dataPath, (d) =>
    EmbeddingDatasetSchema.parse(d),
  )
  const [layout, setLayout] = useState<Layout>('pca')
  const [perplexity, setPerplexity] = useState(30)
  const [highlighted, setHighlighted] = useState<Set<number>>(new Set())
  const { coords, steps, done, run, stop } = useTsne()

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

  // t-SNE runs in a worker: (re)start on entering t-SNE / data / perplexity
  // change; stop on leaving. The worker streams back the evolving solution.
  useEffect(() => {
    if (layout === 'tsne' && data) {
      run(
        data.nodes.map((n) => n.vec),
        { perplexity, maxIter: TSNE_MAX_ITER },
      )
    } else {
      stop()
    }
  }, [layout, data, perplexity, run, stop])

  const tsnePoints = useMemo<ScatterPoint[]>(
    () =>
      data
        ? coords.map((c, i) => ({
            id: data.nodes[i].id,
            community: data.nodes[i].community,
            target: data.nodes[i].target,
            x: c[0],
            y: c[1],
          }))
        : [],
    [data, coords],
  )

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

  const points = layout === 'pca' ? pcaPoints : tsnePoints
  const converging = layout === 'tsne' && !done

  return (
    <Wrapper>
      <ScatterPlot
        points={points}
        getColor={getColor}
        highlightedCommunities={highlighted}
        dictionary={dictionary}
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
            isActive={layout === 'pca'}
            onClick={() => setLayout('pca')}
          />
          <Toggle
            label="t-SNE"
            isActive={layout === 'tsne'}
            onClick={() => setLayout('tsne')}
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
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                run(
                  data.nodes.map((n) => n.vec),
                  { perplexity, maxIter: TSNE_MAX_ITER },
                )
              }
            >
              re-run
            </Button>
            <span className="text-sm tabular-nums text-gray-500">
              {converging
                ? `iterating… ${steps}/${TSNE_MAX_ITER}`
                : `done (${steps})`}
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
  isActive?: boolean
  onClick: () => void
}

function Toggle({
  label,
  isActive = false,
  onClick,
}: ToggleProps): JSX.Element {
  return (
    <Button
      variant={isActive ? 'default' : 'outline'}
      size="sm"
      onClick={onClick}
    >
      {label}
    </Button>
  )
}

const color = d3.scaleOrdinal<number, string>(d3.schemeTableau10)
const NO_COMMUNITY_COLOR = '#cbd5e1' // grey for isolates (community -1)

function getColor(community: number): string {
  return community < 0 ? NO_COMMUNITY_COLOR : color(community)
}
