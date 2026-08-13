import type { Dictionary } from '@lib/build/cedict'
import type { MasterTerm } from '@lib/terms'
import { useState, useEffect, useMemo, type ReactNode } from 'react'
import useData from '@lib/browser/hooks/useData'
import useTsne from '@lib/browser/hooks/useTsne'
import { EmbeddingDatasetSchema, type EmbeddingNode } from '@lib/embeddings'
import {
  groupCommunities,
  COMMUNITY_SORT_LABELS,
  type CommunitySort,
} from '@lib/communities'
import { pinyinKeywords } from '@lib/pinyin'
import Button from './Button'
import Select, { type SelectOption } from './Select'
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

/**
 * Which projection the plot is showing. Besides the pipeline's precomputed
 * layouts (keyed by their own ids), two are computed here: `pca` reads the
 * export's variance-ordered columns, `tsne-live` runs t-SNE in a worker for a
 * perplexity nobody precomputed.
 */
const PCA = 'pca'
const LIVE_TSNE = 'tsne-live'

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

  /**
   * How to order the words listed inside each community, in the legend and the
   * community dialog. Defaults to `proximity` (closeness to that community's
   * virtue), which `notes/community-legend-ordering.md` measures as the most
   * informative; `strength` is the previous behaviour.
   */
  communitySort?: CommunitySort
}

/**
 * Data-loading + controls wrapper (the `MultiNetwork.tsx` of scatter plots).
 * Fetches one reduced-vector dataset and hands the pure `ScatterPlot` a 2-D
 * layout, chosen from three sources:
 *
 * - the pipeline's **precomputed t-SNE** layouts, read straight off the dataset
 *   as x/y — the default, because t-SNE reads the space far better than PCA and
 *   this is the only way to have it on screen immediately;
 * - **PCA**, free from columns 0/1 of the variance-ordered vector;
 * - **live t-SNE** in a worker, for a perplexity the pipeline didn't precompute.
 *
 * Only the coordinates differ — every point carries the same node attributes
 * (community, strength, doc_freq, …) whichever layout is showing.
 */
export default function EmbeddingScatter({
  data: dataPath,
  dictionary,
  stepsPerPost = 2,
  minDocFreq = 0,
  maxDocFreq = 1,
  communitySort = 'prototypicality',
}: EmbeddingScatterProps): JSX.Element {
  const { status, data, errorMessage } = useData(dataPath, (d) =>
    EmbeddingDatasetSchema.parse(d),
  )
  // `null` until the picker is touched, so the dataset's own first precomputed
  // layout can stand in as the default — it only arrives with the data, and
  // seeding state from an effect would fight a user who picked something else.
  const [selectedLayout, setSelectedLayout] = useState<string | null>(null)
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

  // Every precomputed layout, then the two the client can compute itself. The
  // dataset's first layout is the default view; a dataset carrying none (an
  // older export, or `tsne_perplexities = ()`) falls back to PCA.
  const layoutOptions = useMemo<SelectOption[]>(
    () => [
      ...(data?.layouts.map((l) => ({ value: l.id, label: l.label })) || []),
      { value: PCA, label: 'PCA' },
      { value: LIVE_TSNE, label: 't-SNE (live)' },
    ],
    [data],
  )
  const layout = selectedLayout ?? data?.layouts[0]?.id ?? PCA
  const precomputed = data?.layouts.find((l) => l.id === layout)

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

  // Rank members within each community so the words that say what it is come
  // first; alphabetical is noise. See `communitySort`.
  const communities = useMemo<EmbeddingNode[][]>(
    () => (data ? groupCommunities(data.nodes, communitySort) : []),
    [data, communitySort],
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

  // Live t-SNE runs in a worker: (re)start on entering it / on the vectors or
  // perplexity changing; stop on leaving. The worker streams back the evolving
  // solution. `vectors` is a dependency because the solution is positional —
  // filtering the nodes without restarting would slide every point onto the
  // wrong word.
  useEffect(() => {
    if (layout === LIVE_TSNE && data) {
      run(vectors, { perplexity, maxIter: TSNE_MAX_ITER, stepsPerPost })
    } else {
      stop()
    }
  }, [layout, data, vectors, perplexity, stepsPerPost, run, stop])

  const livePoints = useMemo<ScatterPoint[]>(() => {
    // The worker's solution is parallel to what was sent, so a length mismatch
    // means it's mid-restart on a new vocabulary — plot nothing over guessing.
    if (coords.length !== pcaPoints.length) return []
    return pcaPoints.map((p, i) => ({ ...p, x: coords[i][0], y: coords[i][1] }))
  }, [pcaPoints, coords])

  // A precomputed layout is keyed by node id, so filtering is a lookup: a node
  // the layout has no entry for simply isn't plotted.
  const precomputedPoints = useMemo<ScatterPoint[]>(() => {
    if (!precomputed) return []
    return pcaPoints.flatMap((p) => {
      const xy = precomputed.coords[p.id]
      return xy ? [{ ...p, x: xy[0], y: xy[1] }] : []
    })
  }, [pcaPoints, precomputed])

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

  const points = precomputed
    ? precomputedPoints
    : layout === LIVE_TSNE
      ? livePoints
      : pcaPoints
  const converging = layout === LIVE_TSNE && !done

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
              description={`The Louvain communities, with elements sorted by ${COMMUNITY_SORT_LABELS[communitySort]}.`}
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
        <Select
          className="text-sm"
          label="Projection"
          value={layout}
          options={layoutOptions}
          onChange={setSelectedLayout}
        />

        {layout === LIVE_TSNE && (
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
                run(vectors, {
                  perplexity,
                  maxIter: TSNE_MAX_ITER,
                  stepsPerPost,
                })
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
