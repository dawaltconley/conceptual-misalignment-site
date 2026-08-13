import type { Dictionary } from '@lib/build/cedict'
import type { MasterTerm } from '@lib/terms'
import { useState, useEffect, useMemo, type ReactNode } from 'react'
import debounce from 'lodash/debounce'
import useData from '@lib/browser/hooks/useData'
import useTsne from '@lib/browser/hooks/useTsne'
import { EmbeddingDatasetSchema, type EmbeddingNode } from '@lib/embeddings'
import {
  groupCommunities,
  COMMUNITY_SORT_LABELS,
  type CommunitySort,
} from '@lib/communities'
import { pinyinKeywords } from '@lib/pinyin'
import Toggle from './Toggle'
import Checkbox from './Checkbox'
import Progress from './Progress'
import TickRange, { nearest } from './TickRange'
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

type Method = 'tsne' | 'pca'

const TSNE_MAX_ITER = 500
const PERPLEXITY_MIN = 5
const PERPLEXITY_MAX = 50
/** Long enough that dragging the slider doesn't queue a run per pixel. */
const RECOMPUTE_DEBOUNCE_MS = 350

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
 * layout: t-SNE (the default — it reads the space far better than PCA) or PCA,
 * which is free from columns 0/1 of the variance-ordered vector.
 *
 * The t-SNE view has two modes, and the perplexity slider is the same control in
 * both. By default it snaps to the perplexities the pipeline precomputed, whose
 * ticks are drawn under the track: those are instant, and they may be layouts of
 * the *untruncated* vectors, which the browser doesn't have. Ticking
 * "recompute" frees the slider and runs t-SNE here instead, over the reduced
 * vectors that were downloaded — any perplexity, at the cost of a few seconds of
 * iteration.
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
  const [method, setMethod] = useState<Method>('tsne')
  const [recompute, setRecompute] = useState(false)
  // The raw slider position. What the plot uses is `perplexity` below, which is
  // this snapped to a tick unless the client is recomputing — keeping the raw
  // value in state means un-ticking "recompute" lands on a mark rather than
  // wherever the free slider happened to be.
  const [sliderPerplexity, setSliderPerplexity] = useState(30)
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

  // The perplexities the pipeline shipped a layout for — the slider's ticks.
  // Two layouts at one perplexity (`tsne_sources` listing both `reduced` and
  // `full`) would be one tick with the first layout behind it; the pipeline
  // ships one source at a time, so this stays a 1:1 map in practice.
  const byPerplexity = useMemo(() => {
    const map = new Map<number, string>()
    for (const l of data?.layouts ?? []) {
      const p = l.params.perplexity
      if (typeof p === 'number' && !map.has(p)) map.set(p, l.id)
    }
    return map
  }, [data])
  const ticks = useMemo(
    () => [...byPerplexity.keys()].sort((a, b) => a - b),
    [byPerplexity],
  )

  // With nothing precomputed there is nothing to snap to, so the only way to
  // show a t-SNE is to compute it here — the checkbox then has no meaningful
  // off state and says so by being disabled.
  const canSnap = ticks.length > 0
  const live = recompute || !canSnap
  const perplexity = live ? sliderPerplexity : nearest(ticks, sliderPerplexity)
  const precomputed = live
    ? undefined
    : data?.layouts.find((l) => l.id === byPerplexity.get(perplexity))

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

  // Dragging the slider walks through every intermediate perplexity, and each
  // one would otherwise start (and immediately supersede) a worker run. Debounce
  // so only the value the user settles on is computed.
  const debouncedRun = useMemo(
    () => debounce(run, RECOMPUTE_DEBOUNCE_MS),
    [run],
  )
  useEffect(() => () => debouncedRun.cancel(), [debouncedRun])

  // Live t-SNE runs in a worker: (re)start on entering it / on the vectors or
  // perplexity changing; stop on leaving. The worker streams back the evolving
  // solution. `vectors` is a dependency because the solution is positional —
  // filtering the nodes without restarting would slide every point onto the
  // wrong word.
  useEffect(() => {
    if (method === 'tsne' && live && data) {
      debouncedRun(vectors, {
        perplexity,
        maxIter: TSNE_MAX_ITER,
        stepsPerPost,
      })
    } else {
      debouncedRun.cancel()
      stop()
    }
  }, [
    method,
    live,
    data,
    vectors,
    perplexity,
    stepsPerPost,
    debouncedRun,
    stop,
  ])

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

  const points =
    method === 'pca' ? pcaPoints : precomputed ? precomputedPoints : livePoints

  // The three states a client-side run passes through. `queued` is the debounce
  // window plus the worker's start-up: `useTsne` is idle-but-empty there, which
  // is indistinguishable from finished by `done` alone, and it's also when the
  // plot has no points to draw — so it gets the indeterminate bar rather than an
  // unexplained gap.
  const recomputing = method === 'tsne' && live
  const queued = recomputing && done && steps === 0
  const converging = recomputing && !done

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

      <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="flex gap-2">
          <Toggle
            className="text-sm"
            pressed={method === 'tsne'}
            onPressedChange={() => setMethod('tsne')}
          >
            t-SNE
          </Toggle>
          <Toggle
            className="text-sm"
            pressed={method === 'pca'}
            onPressedChange={() => setMethod('pca')}
          >
            PCA
          </Toggle>
        </div>

        {method === 'tsne' && (
          <>
            <label className="flex items-center gap-2 text-sm">
              perplexity
              <TickRange
                className="w-40"
                aria-label="perplexity"
                min={PERPLEXITY_MIN}
                max={PERPLEXITY_MAX}
                value={perplexity}
                onChange={setSliderPerplexity}
                ticks={ticks}
                snap={!live}
              />
              <span className="w-6 tabular-nums">{perplexity}</span>
              {/* Which vectors are actually behind the dots: a precomputed
                  layout may have embedded the untruncated space, which a
                  recompute here cannot reach. */}
              <span className="text-gray-500">
                {precomputed?.params.dims ?? data.dims}-d
              </span>
            </label>

            <Checkbox
              labelClassName="text-sm"
              checked={live}
              disabled={!canSnap}
              onCheckedChange={setRecompute}
            >
              recompute
              {!canSnap && (
                <span className="ml-1 text-gray-500">
                  (nothing precomputed)
                </span>
              )}
            </Checkbox>
          </>
        )}
      </div>

      {/* Only the client-side run has iteration to report; a precomputed layout
          is already converged, and PCA never iterates. */}
      {recomputing && (
        <Progress
          className="mt-3"
          max={TSNE_MAX_ITER}
          // Indeterminate while queued — there is no step count to honour yet,
          // and a bar pinned at 0% would read as stalled.
          value={queued ? undefined : steps}
          showValue={!queued}
          label={
            queued
              ? 'recomputing…'
              : converging
                ? `iterating — ${steps}/${TSNE_MAX_ITER}`
                : `converged — ${steps} iterations`
          }
        />
      )}
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
    size: Math.log2(data.strength + 1), // works well
    // size: Math.log10(data.pagerank + 1) * 5000,
    data,
  }
}
