import type { Dictionary } from '@lib/build/cedict'
import { useState, useMemo, useId, type ReactNode } from 'react'
import clsx from 'clsx'
import useData from '@lib/browser/hooks/useData'
import { EmbeddingDatasetSchema, type EmbeddingDataset } from '@lib/embeddings'
import { procrustes, applyRotation, crossCovSVD, pca2d } from '@lib/align'
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

// Which space is pinned. 'neutral' pins neither (both rotate to a shared frame).
type Frame = 'chinese' | 'english' | 'neutral'

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

  // 2-D coordinates per corpus, depending on the chosen frame:
  //  - chinese: Chinese pinned to its PCA cols 0/1; English rotated onto it.
  //  - english: English pinned; Chinese rotated onto it.
  //  - neutral: send English through U and Chinese through V (the SVD middle
  //    frame), then a joint PCA of the merged cloud → shared 2-D axes.
  // With no anchors each side falls back to its own cols 0/1 (un-aligned).
  const layout = useMemo<{ zh2d: number[][]; en2d: number[][] }>(() => {
    if (!chineseData || !englishData) return { zh2d: [], en2d: [] }
    const zhAll = chineseData.nodes.map((n) => n.vec)
    const enAll = englishData.nodes.map((n) => n.vec)
    const valid = anchors.filter((a) => enVecs.has(a.en) && zhVecs.has(a.zh))
    const enA = valid.map((a) => enVecs.get(a.en)!)
    const zhA = valid.map((a) => zhVecs.get(a.zh)!)
    const cols01 = (rows: number[][]) => rows.map((r) => [r[0] ?? 0, r[1] ?? 0])

    if (frame === 'chinese') {
      const en2d =
        valid.length >= 1
          ? cols01(applyRotation(enAll, procrustes(enA, zhA)))
          : cols01(enAll)
      return { zh2d: cols01(zhAll), en2d }
    }
    if (frame === 'english') {
      const zh2d =
        valid.length >= 1
          ? cols01(applyRotation(zhAll, procrustes(zhA, enA)))
          : cols01(zhAll)
      return { zh2d, en2d: cols01(enAll) }
    }
    // neutral
    let zhT = zhAll
    let enT = enAll
    if (valid.length >= 1) {
      const { u, v } = crossCovSVD(enA, zhA)
      enT = applyRotation(enAll, u)
      zhT = applyRotation(zhAll, v)
    }
    const joint = pca2d([...zhT, ...enT])
    return {
      zh2d: joint.slice(0, zhAll.length),
      en2d: joint.slice(zhAll.length),
    }
  }, [chineseData, englishData, anchors, enVecs, zhVecs, frame])

  const points = useMemo<ScatterPoint[]>(() => {
    if (!chineseData || !englishData) return []
    const zhPts = chineseData.nodes.map<ScatterPoint>((n, i) => ({
      id: n.id,
      x: layout.zh2d[i]?.[0] ?? 0,
      y: layout.zh2d[i]?.[1] ?? 0,
      community: CHINESE,
      target: n.target,
    }))
    const enPts = englishData.nodes.map<ScatterPoint>((n, i) => ({
      id: n.id,
      x: layout.en2d[i]?.[0] ?? 0,
      y: layout.en2d[i]?.[1] ?? 0,
      community: ENGLISH,
      target: n.target,
    }))
    return [...zhPts, ...enPts]
  }, [chineseData, englishData, layout])

  const role = (side: Frame): string =>
    frame === 'neutral' ? ' — aligned' : frame === side ? ' — fixed' : ' — aligned'
  const legend: LegendLabel[] = [
    { id: String(CHINESE), color: getColor(CHINESE), description: `Mengzi (Chinese)${role('chinese')}` },
    { id: String(ENGLISH), color: getColor(ENGLISH), description: `SEP (English)${role('english')}` },
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

      <div className="mt-3 flex items-center gap-2 text-sm">
        <span className="text-gray-600">Frame:</span>
        {(['chinese', 'english', 'neutral'] as Frame[]).map((f) => (
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
