import type { Dictionary } from '@lib/build/cedict'
import type { MasterTerm } from '@lib/terms'
import { useState, useEffect, useMemo, type ReactNode } from 'react'
import useData from '@lib/browser/hooks/useData'
import useTsne from '@lib/browser/hooks/useTsne'
import { EmbeddingDatasetSchema, type EmbeddingNode } from '@lib/embeddings'
import { pinyinKeywords } from '@lib/pinyin'
import Button from './Button'
import Toggle from './Toggle'
import TagsCombobox from './TagsCombobox'
import type { ComboboxOption } from './Combobox'
import {
  ScatterSkeleton,
  Wrapper as ScatterWrapper,
  type ScatterPoint,
} from './ScatterPlot'
import ScatterPlot from './CanvasScatterPlot'
import ScatterLegend, { type LegendLabel } from './ScatterLegend'
import CommunityDialog from './CommunityDialog'
import { Dialog } from '@base-ui/react'
import * as d3 from 'd3'

type Layout = 'pca' | 'tsne'

const TSNE_MAX_ITER = 500

interface EmbeddingScatterProps {
  terms: MasterTerm[]
  data: string
  dictionary?: Dictionary

  /**
   * The number of steps in the t-SNE calculation that are calculated before
   * the component animates a new frame.
   */
  stepsPerPost?: number

  /**
   * The minimum proportion of documents a term must appear in to be shown in
   * the scatterplot. Should be a number between 0 and 1.
   */
  minDocFreq?: number

  /**
   * The maximum proportion of documents a term can appear in and be shown in
   * the scatterplot. Should be a number between 0 and 1.
   */
  maxDocFreq?: number
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
  stepsPerPost = 2,
  minDocFreq = 0,
  maxDocFreq = 1,
}: EmbeddingScatterProps): JSX.Element {
  const { status, data, errorMessage } = useData(dataPath, (d) =>
    EmbeddingDatasetSchema.parse(d),
  )
  const [layout, setLayout] = useState<Layout>('pca')
  const [perplexity, setPerplexity] = useState(30)
  const [highlighted, setHighlighted] = useState<Set<number>>(new Set())
  // `null` until the control is touched, which is what lets the corpus's own
  // targets stand in as the initial selection: they only arrive with the data,
  // and seeding state from an effect would fight a user who deselects them.
  const [selectedTargets, setSelectedTargets] = useState<string[] | null>(null)
  const { coords, steps, done, run, stop } = useTsne()

  // The corpus's core terms — the pipeline's `target` nodes. They seed the
  // selection rather than being pinned to it, so any of them can be dropped.
  const coreTargets = useMemo<string[]>(
    () => data?.nodes.filter((n) => n.target).map((n) => n.id) || [],
    [data],
  )
  // Whatever is selected *is* what the plot emphasises: larger, outlined, labelled.
  const selection = selectedTargets ?? coreTargets
  const targets = useMemo<Set<string>>(() => new Set(selection), [selection])

  const filteredNodes = useMemo<EmbeddingNode[]>(
    () =>
      data?.nodes.filter((n) => {
        if (targets.has(n.id)) return true
        const docFreq = n.doc_freq / data.documents
        return docFreq > minDocFreq && docFreq <= maxDocFreq
      }) || [],
    [data, targets],
  )

  // PCA layout — free: the reduced columns are variance-ordered, so [0],[1]
  // are the 2-D principal-component coordinates.
  const pcaPoints = useMemo<ScatterPoint[]>(
    () =>
      data
        ? filteredNodes.map((n) => nodeToScatterPoint(n, data?.documents))
        : [],
    [filteredNodes, data],
  )

  const vectors = useMemo<number[][]>(
    () => filteredNodes.map((n) => n.vec),
    [filteredNodes],
  )

  // Every node is selectable. Chinese options carry pinyin so the vocabulary is
  // reachable from a plain keyboard — `ren2`, `ren` and `rén` all find 仁.
  const options = useMemo<ComboboxOption[]>(
    () =>
      data?.nodes.map((n) => {
        const entry = dictionary?.[n.id]
        const pinyin = entry?.readings[0]?.pinyin
        return {
          value: n.id,
          keywords: pinyin ? pinyinKeywords(pinyin) : undefined,
          note: pinyin,
        }
      }) || [],
    [data, dictionary],
  )

  // Rank members by weighted degree (strength) so the tightest/most
  // central words in each community come first; alphabetical is noise.
  const communities = useMemo<EmbeddingNode[][]>(
    () =>
      data?.nodes
        .toSorted((a, b) => b.strength - a.strength)
        .reduce<EmbeddingNode[][]>((communities, n) => {
          communities[n.community] ??= []
          communities[n.community].push(n)
          return communities
        }, []) || [],
    [data],
  )
  const communityLegend = useMemo<LegendLabel[]>(
    () =>
      communities.map<LegendLabel>((nodes, community) => ({
        id: community.toString(),
        color: getColor(community),
        description: `C${community}: ${nodes.map((n) => n.id).join(', ')}`,
        dialog: Dialog.createHandle(),
      })),
    [communities],
  )

  // t-SNE runs in a worker: (re)start on entering t-SNE / data / perplexity
  // change; stop on leaving. The worker streams back the evolving solution.
  useEffect(() => {
    if (layout === 'tsne' && data) {
      run(vectors, { perplexity, maxIter: TSNE_MAX_ITER, stepsPerPost })
    } else {
      stop()
    }
  }, [layout, data, perplexity, run, stop])

  const tsnePoints = useMemo<ScatterPoint[]>(() => {
    if (!coords || !coords.length) return []
    return pcaPoints.map((p, i) => ({
      ...p,
      x: coords[i][0],
      y: coords[i][1],
    }))
  }, [data, coords])

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
        targets={targets}
        isHighlighted={(p) => highlighted.has(Number(p.data?.community))}
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

      {communityLegend.map(({ id, dialog }) => {
        const nodes = communities[Number(id)]
        return (
          nodes && (
            <CommunityDialog
              key={id}
              title={`Louvain community C${id}`}
              description="The Louvain communities, with elements sorted by embedding strength."
              nodes={nodes}
              handle={dialog}
              dictionary={dictionary}
            />
          )
        )
      })}

      <TagsCombobox
        className="mt-4"
        label="Highlighted terms"
        value={selection}
        options={options}
        onChange={setSelectedTargets}
        placeholder={selection.length ? '' : 'add a term…'}
      />

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <div className="flex gap-2">
          <Toggle
            className="text-sm"
            pressed={layout === 'pca'}
            onPressedChange={() => setLayout('pca')}
          >
            PCA
          </Toggle>
          <Toggle
            className="text-sm"
            pressed={layout === 'tsne'}
            onPressedChange={() => setLayout('tsne')}
          >
            t-SNE
          </Toggle>
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
              className="text-sm"
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

const color = d3.scaleOrdinal<number, string>(d3.schemeTableau10)
const NO_COMMUNITY_COLOR = '#cbd5e1' // grey for isolates (community -1)

function getColor(community: number): string {
  return community < 0 ? NO_COMMUNITY_COLOR : color(community)
}

function nodeToScatterPoint(
  { id, vec, ...data }: EmbeddingNode,
  corpusDocuments: number,
): ScatterPoint {
  return {
    id,
    x: vec[0] ?? 0,
    y: vec[1] ?? 0,
    color: getColor(data.community),
    opacity: data.doc_freq / corpusDocuments,
    size: Math.log2(data.strength + 1) * 2, // works well
    // size: Math.log10(data.pagerank + 1) * 5000,
    data,
  }
}
