import type { Dictionary } from '@lib/build/cedict'
import { useState, useMemo, useId, type ReactNode } from 'react'
import clsx from 'clsx'
import useData from '@lib/browser/hooks/useData'
import { EmbeddingDatasetSchema, type EmbeddingDataset } from '@lib/embeddings'
import {
  procrustes,
  applyRotation,
  symmetricRotations,
  pca2d,
} from '@lib/align'
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
  const [zhInput, setZhInput] = useState('')
  const [enInput, setEnInput] = useState('')
  const [frame, setFrame] = useState<Frame>('chinese')
  const listId = useId()

  const chineseData = zh.status === 'success' ? zh.data : null
  const englishData = en.status === 'success' ? en.data : null

  const zhVecs = useMemo(() => vecMap(chineseData), [chineseData])
  const enVecs = useMemo(() => vecMap(englishData), [englishData])

  // 2-D coordinates for both corpora under the selected frame. `chinese`/`english`
  // hold one space fixed (its PCA cols 0/1) and rotate the other onto it via
  // one-sided Procrustes; `neutral` rotates both into a shared frame and re-projects
  // with a joint PCA. Falls back to un-aligned cols 0/1 when there are no anchors.
  const layout = useMemo<{ zh: number[][]; en: number[][] }>(() => {
    if (!chineseData || !englishData) return { zh: [], en: [] }
    const zhRaw = chineseData.nodes.map((n) => n.vec)
    const enRaw = englishData.nodes.map((n) => n.vec)
    const cols01 = (v: number[]) => [v[0] ?? 0, v[1] ?? 0]

    const valid = anchors.filter((a) => enVecs.has(a.en) && zhVecs.has(a.zh))
    const A = valid.map((a) => enVecs.get(a.en)!) // English anchor rows
    const B = valid.map((a) => zhVecs.get(a.zh)!) // Chinese anchor rows

    if (frame === 'chinese') {
      const en = valid.length ? applyRotation(enRaw, procrustes(A, B)) : enRaw
      return { zh: zhRaw.map(cols01), en: en.map(cols01) }
    }
    if (frame === 'english') {
      const zh = valid.length ? applyRotation(zhRaw, procrustes(B, A)) : zhRaw
      return { zh: zh.map(cols01), en: enRaw.map(cols01) }
    }
    // neutral
    if (!valid.length) return { zh: zhRaw.map(cols01), en: enRaw.map(cols01) }
    const { left, right } = symmetricRotations(A, B)
    const enRot = applyRotation(enRaw, left)
    const zhRot = applyRotation(zhRaw, right)
    const coords = pca2d([...zhRot, ...enRot])
    return { zh: coords.slice(0, zhRaw.length), en: coords.slice(zhRaw.length) }
  }, [chineseData, englishData, anchors, enVecs, zhVecs, frame])

  const points = useMemo<ScatterPoint[]>(() => {
    if (!chineseData || !englishData) return []
    const zhPts = chineseData.nodes.map<ScatterPoint>((n, i) => ({
      id: n.id,
      x: layout.zh[i]?.[0] ?? 0,
      y: layout.zh[i]?.[1] ?? 0,
      community: CHINESE,
      target: n.target,
    }))
    const enPts = englishData.nodes.map<ScatterPoint>((n, i) => ({
      id: n.id,
      x: layout.en[i]?.[0] ?? 0,
      y: layout.en[i]?.[1] ?? 0,
      community: ENGLISH,
      target: n.target,
    }))
    return [...zhPts, ...enPts]
  }, [chineseData, englishData, layout])

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

  const zhHas = zhVecs.has(zhInput)
  const enHas = enVecs.has(enInput)
  const canAdd =
    zhHas && enHas && !anchors.some((a) => a.zh === zhInput && a.en === enInput)

  const addAnchor = () => {
    if (!canAdd) return
    setAnchors((a) => [...a, { zh: zhInput, en: enInput }])
    setZhInput('')
    setEnInput('')
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

      <div className="mt-3 flex items-center gap-2 text-sm">
        <span className="font-bold">Frame:</span>
        {FRAMES.map((f) => (
          <button
            key={f}
            onClick={() => setFrame(f)}
            className={clsx(
              'rounded border border-gray-900 px-2 py-0.5 capitalize duration-150 hover:bg-red-200',
              frame === f && 'border-red-500 bg-red-500 text-white',
            )}
          >
            {f}
          </button>
        ))}
      </div>

      <div className="mt-4">
        <p className="mb-1 text-sm font-bold">
          Anchor pairs{' '}
          <span className="font-normal text-gray-500">
            (need several for a meaningful alignment)
          </span>
        </p>

        <div className="flex flex-wrap items-end gap-2">
          <Field
            label="Chinese term"
            value={zhInput}
            onChange={setZhInput}
            listId={`${listId}-zh`}
            valid={!zhInput || zhHas}
          />
          <Field
            label="English term"
            value={enInput}
            onChange={setEnInput}
            listId={`${listId}-en`}
            valid={!enInput || enHas}
          />
          <button
            disabled={!canAdd}
            onClick={addAnchor}
            className="rounded border border-gray-900 px-2 py-1 text-sm duration-150 enabled:hover:bg-red-200 disabled:opacity-40"
          >
            + add
          </button>
        </div>

        <datalist id={`${listId}-zh`}>
          {chineseData.nodes.map((n) => (
            <option key={n.id} value={n.id} />
          ))}
        </datalist>
        <datalist id={`${listId}-en`}>
          {englishData.nodes.map((n) => (
            <option key={n.id} value={n.id} />
          ))}
        </datalist>

        {anchors.length > 0 && (
          <ul className="mt-3 flex flex-wrap gap-2">
            {anchors.map((a, i) => (
              <li
                key={`${a.zh}-${a.en}`}
                className="flex items-center gap-1 rounded bg-gray-100 px-2 py-0.5 text-sm ring-1 ring-gray-300"
              >
                <span>
                  {a.zh} ↔ {a.en}
                </span>
                <button
                  aria-label={`remove ${a.zh} ↔ ${a.en}`}
                  className="text-gray-500 hover:text-red-600"
                  onClick={() =>
                    setAnchors((prev) => prev.filter((_, j) => j !== i))
                  }
                >
                  ×
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

interface FieldProps {
  label: string
  value: string
  onChange: (v: string) => void
  listId: string
  valid: boolean
}

function Field({
  label,
  value,
  onChange,
  listId,
  valid,
}: FieldProps): ReactNode {
  return (
    <label className="flex flex-col text-sm">
      <span className="text-gray-600">{label}</span>
      <input
        type="text"
        list={listId}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={clsx(
          'rounded border px-2 py-1',
          valid ? 'border-gray-900' : 'border-red-500 bg-red-50',
        )}
      />
    </label>
  )
}
