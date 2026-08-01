import type { Dictionary } from '@lib/build/cedict'
import { useState, useEffect, useMemo } from 'react'
import useData from '@lib/browser/hooks/useData'
import useTsne from '@lib/browser/hooks/useTsne'
import { EmbeddingDatasetSchema, type EmbeddingDataset } from '@lib/embeddings'
import {
  procrustes,
  applyRotation,
  symmetricRotations,
  pca2d,
} from '@lib/align'
import { pinyinKeywords } from '@lib/pinyin'
import Button from './Button'
import Toggle from './Toggle'
import Combobox, { type ComboboxOption } from './Combobox'
import Icon from './Icon'
import { faXmark } from '@fortawesome/pro-regular-svg-icons/faXmark'
import CanvasScatterPlot from './CanvasScatterPlot'
import { ScatterSkeleton, type ScatterPoint } from './ScatterPlot'
import ScatterLegend, { type LegendLabel } from './ScatterLegend'

// Corpus is encoded in `community`: 0 = Chinese, 1 = English.
const CHINESE = 0
const ENGLISH = 1

// Which space stays fixed while the other rotates onto it. `neutral` rotates both
// into a shared frame (symmetric Procrustes) and re-projects with a joint PCA.
const FRAMES = ['chinese', 'english', 'neutral'] as const
type Frame = (typeof FRAMES)[number]
const FRAME_LABEL: Record<Frame, { zh: string; en: string }> = {
  chinese: { zh: 'Mengzi — fixed frame', en: 'SEP — aligned' },
  english: { zh: 'Mengzi — aligned', en: 'SEP — fixed frame' },
  neutral: { zh: 'Mengzi — aligned', en: 'SEP — aligned' },
}

const METHODS = ['pca', 'tsne'] as const
type Method = (typeof METHODS)[number]
const METHOD_LABEL: Record<Method, string> = { pca: 'PCA', tsne: 't-SNE' }
const TSNE_MAX_ITER = 500
const CORPUS_COLOR: Record<number, string> = {
  [CHINESE]: '#d62728', // red
  [ENGLISH]: '#1f77b4', // blue
}
const getColor = (community: number): string =>
  CORPUS_COLOR[community] ?? '#cbd5e1'

interface Anchor {
  zh: string
  en: string
}

interface AlignmentScatterProps {
  /** Path to the Chinese reduced-vector dataset (the fixed frame). */
  chinese: string
  /** Path to the English reduced-vector dataset (aligned onto the Chinese frame). */
  english: string
  dictionary?: Dictionary
}

/**
 * Cross-lingual alignment view. The Chinese space is the fixed backdrop (its
 * PCA columns 0/1); the English space is rotated onto it by an orthogonal
 * Procrustes map fit on user-chosen anchor pairs, then read in the same columns.
 * Anchors are chosen entirely client-side (two autocomplete fields), so the user
 * can watch how anchor choice reshapes the shared space — which *is* the research
 * question. Reuses `CanvasScatterPlot` for rendering.
 */
export default function AlignmentScatter({
  chinese: chinesePath,
  english: englishPath,
  dictionary,
}: AlignmentScatterProps): JSX.Element {
  const zh = useData(chinesePath, (d) => EmbeddingDatasetSchema.parse(d))
  const en = useData(englishPath, (d) => EmbeddingDatasetSchema.parse(d))

  const [anchors, setAnchors] = useState<Anchor[]>([])
  const [zhInput, setZhInput] = useState<string | null>(null)
  const [enInput, setEnInput] = useState<string | null>(null)
  const [frame, setFrame] = useState<Frame>('chinese')
  const [method, setMethod] = useState<Method>('pca')
  const [perplexity, setPerplexity] = useState(30)

  const chineseData = zh.status === 'success' ? zh.data : null
  const englishData = en.status === 'success' ? en.data : null

  // Anchor options. The Chinese side carries pinyin so the whole vocabulary is
  // reachable from a plain keyboard — `ren2`, `ren` and `rén` all find 仁.
  const zhOptions = useMemo<ComboboxOption[]>(
    () =>
      (chineseData?.nodes ?? []).map((n) => {
        const entry = dictionary?.[n.id]
        return {
          value: n.id,
          keywords: entry && pinyinKeywords(entry.pinyin),
          note: entry?.pinyin,
        }
      }),
    [chineseData, dictionary],
  )
  const enOptions = useMemo<ComboboxOption[]>(
    () => (englishData?.nodes ?? []).map((n) => ({ value: n.id })),
    [englishData],
  )

  const zhVecs = useMemo(() => vecMap(chineseData), [chineseData])
  const enVecs = useMemo(() => vecMap(englishData), [englishData])

  // Full shared-frame vectors for both corpora — the input to *both* projections.
  // `chinese`/`english` hold one space fixed and rotate the other onto it via
  // one-sided Procrustes; `neutral` rotates both into the SVD's shared frame.
  // With no anchors, vectors are left un-aligned (each in its own basis).
  const aligned = useMemo<{ zh: number[][]; en: number[][] }>(() => {
    if (!chineseData || !englishData) return { zh: [], en: [] }
    const zhRaw = chineseData.nodes.map((n) => n.vec)
    const enRaw = englishData.nodes.map((n) => n.vec)
    const valid = anchors.filter((a) => enVecs.has(a.en) && zhVecs.has(a.zh))
    if (!valid.length) return { zh: zhRaw, en: enRaw }
    const A = valid.map((a) => enVecs.get(a.en)!) // English anchor rows
    const B = valid.map((a) => zhVecs.get(a.zh)!) // Chinese anchor rows
    if (frame === 'chinese')
      return { zh: zhRaw, en: applyRotation(enRaw, procrustes(A, B)) }
    if (frame === 'english')
      return { zh: applyRotation(zhRaw, procrustes(B, A)), en: enRaw }
    const { left, right } = symmetricRotations(A, B)
    return { zh: applyRotation(zhRaw, right), en: applyRotation(enRaw, left) }
  }, [chineseData, englishData, anchors, enVecs, zhVecs, frame])

  const zhCount = chineseData?.nodes.length ?? 0
  const combined = useMemo(() => [...aligned.zh, ...aligned.en], [aligned])

  // PCA projection: fixed frames read cols 0/1 (their own PCA axes); neutral needs
  // a joint PCA since the shared singular frame isn't variance-ordered.
  const pcaCoords = useMemo<number[][]>(
    () =>
      frame === 'neutral'
        ? pca2d(combined)
        : combined.map((v) => [v[0] ?? 0, v[1] ?? 0]),
    [combined, frame],
  )

  // t-SNE projection: a joint embedding of the combined shared-frame vectors,
  // computed in a worker and recomputed whenever the alignment changes (any
  // anchor/frame edit), so well-aligned pairs migrate together as anchors are added.
  const { coords: tsneCoords, run: runTsne, stop: stopTsne } = useTsne()
  useEffect(() => {
    if (method === 'tsne' && combined.length >= 2) {
      runTsne(combined, { perplexity, maxIter: TSNE_MAX_ITER, stepsPerPost: 1 })
    } else {
      stopTsne()
    }
  }, [method, combined, perplexity, runTsne, stopTsne])

  const coords = method === 'pca' ? pcaCoords : tsneCoords
  const points = useMemo<ScatterPoint[]>(() => {
    if (!chineseData || !englishData) return []
    return coords.map<ScatterPoint>((c, i) => {
      const node =
        i < zhCount ? chineseData.nodes[i] : englishData.nodes[i - zhCount]
      return {
        id: node.id,
        x: c?.[0] ?? 0,
        y: c?.[1] ?? 0,
        community: i < zhCount ? CHINESE : ENGLISH,
        target: node.target,
      }
    })
  }, [chineseData, englishData, coords, zhCount])

  const legend: LegendLabel[] = [
    {
      id: String(CHINESE),
      color: getColor(CHINESE),
      description: FRAME_LABEL[frame].zh,
    },
    {
      id: String(ENGLISH),
      color: getColor(ENGLISH),
      description: FRAME_LABEL[frame].en,
    },
  ]

  if (zh.status === 'error' || en.status === 'error') {
    return (
      <div className="flex min-h-96 items-center justify-center">
        Error: {zh.errorMessage ?? en.errorMessage}
      </div>
    )
  }
  if (!chineseData || !englishData) return <ScatterSkeleton />

  // The comboboxes only emit values that exist in their lists, so the pair is
  // valid as soon as both sides are set.
  const canAdd =
    zhInput !== null &&
    enInput !== null &&
    !anchors.some((a) => a.zh === zhInput && a.en === enInput)

  const addAnchor = () => {
    if (!canAdd) return
    setAnchors((a) => [...a, { zh: zhInput, en: enInput }])
    setZhInput(null)
    setEnInput(null)
  }

  return (
    <div className="w-full">
      <CanvasScatterPlot
        points={points}
        getColor={getColor}
        dictionary={dictionary}
      />

      <div className="ml-8 mt-2 text-xs">
        <ScatterLegend labels={legend} />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
        <div className="flex items-center gap-2">
          <span className="font-bold">Frame:</span>
          {FRAMES.map((f) => (
            <Toggle
              key={f}
              pressed={frame === f}
              onPressedChange={() => setFrame(f)}
              className="capitalize"
            >
              {f}
            </Toggle>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <span className="font-bold">Projection:</span>
          {METHODS.map((m) => (
            <Toggle
              key={m}
              pressed={method === m}
              onPressedChange={() => setMethod(m)}
            >
              {METHOD_LABEL[m]}
            </Toggle>
          ))}
          {method === 'tsne' && (
            <label className="flex items-center gap-2">
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
          )}
        </div>
      </div>

      <div className="mt-4">
        <p className="mb-1 text-sm font-bold">
          Anchor pairs{' '}
          <span className="font-normal text-gray-500">
            (need several for a meaningful alignment)
          </span>
        </p>

        <div className="flex flex-wrap items-center gap-2">
          <Combobox
            label="Chinese term"
            value={zhInput}
            options={zhOptions}
            onChange={setZhInput}
            placeholder="hanzi or pinyin…"
            controlClassName="w-52"
          />
          <Combobox
            label="English term"
            value={enInput}
            options={enOptions}
            onChange={setEnInput}
            placeholder="term…"
            controlClassName="w-52"
          />
          <Button disabled={!canAdd} onClick={addAnchor} className="text-sm">
            + add
          </Button>
        </div>

        {anchors.length > 0 && (
          <ul className="mt-3 flex flex-wrap gap-2">
            {anchors.map((a, i) => (
              <li key={`${a.zh}-${a.en}`} className="chip">
                <span>
                  {a.zh} ↔ {a.en}
                </span>
                <button
                  aria-label={`Remove ${a.zh} ↔ ${a.en}`}
                  className="chip__remove"
                  onClick={() =>
                    setAnchors((prev) => prev.filter((_, j) => j !== i))
                  }
                >
                  <Icon icon={faXmark} width="0.75em" height="0.75em" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

function vecMap(data: EmbeddingDataset | null): Map<string, number[]> {
  return new Map(data ? data.nodes.map((n) => [n.id, n.vec]) : [])
}
