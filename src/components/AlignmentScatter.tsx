import type { Dictionary } from '@lib/build/cedict'
import { useState, useMemo, useId, type ReactNode } from 'react'
import clsx from 'clsx'
import useData from '@lib/browser/hooks/useData'
import { EmbeddingDatasetSchema, type EmbeddingDataset } from '@lib/embeddings'
import { procrustes, applyRotation } from '@lib/align'
import CanvasScatterPlot from './CanvasScatterPlot'
import { ScatterSkeleton, type ScatterPoint } from './ScatterPlot'
import ScatterLegend, { type LegendLabel } from './ScatterLegend'

// Corpus is encoded in `community`: 0 = Chinese (fixed frame), 1 = English (aligned).
const CHINESE = 0
const ENGLISH = 1
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
  const listId = useId()

  const chineseData = zh.status === 'success' ? zh.data : null
  const englishData = en.status === 'success' ? en.data : null

  const zhVecs = useMemo(() => vecMap(chineseData), [chineseData])
  const enVecs = useMemo(() => vecMap(englishData), [englishData])

  // Fit Procrustes on the valid anchors and rotate every English vector into the
  // Chinese space. With no anchors, fall back to identity (English un-aligned).
  const alignedEnglish = useMemo<number[][]>(() => {
    if (!englishData) return []
    const raw = englishData.nodes.map((n) => n.vec)
    const valid = anchors.filter((a) => enVecs.has(a.en) && zhVecs.has(a.zh))
    if (valid.length < 1) return raw
    const A = valid.map((a) => enVecs.get(a.en)!)
    const B = valid.map((a) => zhVecs.get(a.zh)!)
    return applyRotation(raw, procrustes(A, B))
  }, [englishData, anchors, enVecs, zhVecs])

  const points = useMemo<ScatterPoint[]>(() => {
    if (!chineseData || !englishData) return []
    const zhPts = chineseData.nodes.map<ScatterPoint>((n) => ({
      id: n.id,
      x: n.vec[0] ?? 0,
      y: n.vec[1] ?? 0,
      community: CHINESE,
      target: n.target,
    }))
    const enPts = englishData.nodes.map<ScatterPoint>((n, i) => ({
      id: n.id,
      x: alignedEnglish[i]?.[0] ?? 0,
      y: alignedEnglish[i]?.[1] ?? 0,
      community: ENGLISH,
      target: n.target,
    }))
    return [...zhPts, ...enPts]
  }, [chineseData, englishData, alignedEnglish])

  const legend: LegendLabel[] = [
    { id: String(CHINESE), color: getColor(CHINESE), description: 'Mengzi (Chinese) — fixed frame' },
    { id: String(ENGLISH), color: getColor(ENGLISH), description: 'SEP (English) — aligned' },
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
    zhHas &&
    enHas &&
    !anchors.some((a) => a.zh === zhInput && a.en === enInput)

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

      <div className="mt-4">
        <p className="mb-1 text-sm font-bold">
          Anchor pairs{' '}
          <span className="font-normal text-gray-500">
            (need several for a meaningful alignment)
          </span>
        </p>

        <div className="flex flex-wrap items-end gap-2">
          <Field label="Chinese term" value={zhInput} onChange={setZhInput}
                 listId={`${listId}-zh`} valid={!zhInput || zhHas} />
          <Field label="English term" value={enInput} onChange={setEnInput}
                 listId={`${listId}-en`} valid={!enInput || enHas} />
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

function Field({ label, value, onChange, listId, valid }: FieldProps): ReactNode {
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
