import { useEffect, useState } from 'react'
import Button from './Button'
import Select from './Select'
import Toggle from './Toggle'
import Progress from './Progress'
import Combobox, { type ComboboxOption } from './Combobox'
import TagsCombobox from './TagsCombobox'

const PROGRESS_MAX = 500

interface ComponentGalleryProps {
  /** Hanzi from the Mengzi networks, keyed for pinyin search. */
  hanzi: ComboboxOption[]
  /** The English renderings from the master term index. */
  renderings: string[]
  /** The active Chinese terms. */
  terms: string[]
}

/**
 * Dev-only scratch page for the shared controls (`/gallery`), pointed at real
 * corpus data so pinyin search is exercised against actual CC-CEDICT readings
 * rather than a fixture. Delete along with `pages/gallery.astro` once these
 * are wired into the real views.
 */
export default function ComponentGallery({
  hanzi,
  renderings,
  terms,
}: ComponentGalleryProps): JSX.Element {
  const [term, setTerm] = useState(terms[0] ?? '')
  const [rendering, setRendering] = useState(renderings[0] ?? '')
  const [layout, setLayout] = useState<'pca' | 'tsne'>('pca')
  const [centered, setCentered] = useState(false)
  const [steps, setSteps] = useState(0)
  const [running, setRunning] = useState(false)
  const [pick, setPick] = useState<string | null>(null)
  const [anchors, setAnchors] = useState<string[]>([])

  useEffect(() => {
    if (!running) return
    const id = setInterval(() => {
      setSteps((s) => {
        if (s >= PROGRESS_MAX) {
          setRunning(false)
          return s
        }
        return s + 10
      })
    }, 80)
    return () => clearInterval(id)
  }, [running])

  return (
    <div className="flex flex-col gap-10">
      <Section
        title="Select"
        note="Base UI Select. Same props as the TermNetwork dropdowns it replaced."
      >
        <Select
          label="Chinese term"
          value={term}
          options={terms}
          onChange={setTerm}
          triggerClassName="text-lg"
        />
        <Select
          label="English rendering"
          value={rendering}
          options={renderings}
          onChange={setRendering}
        />
      </Section>

      <Section
        title="Toggle"
        note="Base UI Toggle on the shared .btn styles. The pair is radio-like; the third is a plain on/off."
      >
        <div className="flex gap-2">
          <Toggle
            pressed={layout === 'pca'}
            onPressedChange={() => setLayout('pca')}
          >
            PCA
          </Toggle>
          <Toggle
            pressed={layout === 'tsne'}
            onPressedChange={() => setLayout('tsne')}
          >
            t-SNE
          </Toggle>
        </div>
        <Toggle pressed={centered} onPressedChange={setCentered}>
          mean-centered
        </Toggle>
        <span className="text-sm text-gray-500">
          layout: {layout}, centered: {String(centered)}
        </span>
      </Section>

      <Section
        title="Progress"
        note="Determinate in value/max units, plus the indeterminate sweep."
      >
        <div className="flex w-full max-w-md flex-col gap-4">
          <Progress
            label={`iterating… ${steps}/${PROGRESS_MAX}`}
            value={steps}
            max={PROGRESS_MAX}
            showValue
          />
          <Progress label="waiting on the worker" />
          <div className="flex gap-2">
            <Button className="text-sm" onClick={() => setRunning((r) => !r)}>
              {running ? 'pause' : 'run'}
            </Button>
            <Button
              className="text-sm"
              onClick={() => {
                setRunning(false)
                setSteps(0)
              }}
            >
              reset
            </Button>
          </div>
        </div>
      </Section>

      <Section
        title="Combobox"
        note={`${hanzi.length} hanzi from the Mengzi networks. Try "ren2", "ren" or "rén" — all three find 仁.`}
      >
        <Combobox
          label="Chinese term"
          value={pick}
          options={hanzi}
          onChange={setPick}
          placeholder="hanzi or pinyin…"
          controlClassName="w-64"
        />
        <span className="text-sm text-gray-500">selected: {pick ?? '—'}</span>
      </Section>

      <Section
        title="TagsCombobox"
        note="Same list, multi-select. Backspace on an empty field targets the last tag."
      >
        <div className="w-full max-w-md">
          <TagsCombobox
            label="Alignment anchors"
            value={anchors}
            options={hanzi}
            onChange={setAnchors}
            placeholder="hanzi or pinyin…"
          />
        </div>
        <span className="text-sm text-gray-500">
          {anchors.length ? anchors.join(', ') : 'none selected'}
        </span>
      </Section>
    </div>
  )
}

interface SectionProps {
  title: string
  note: string
  children: React.ReactNode
}

function Section({ title, note, children }: SectionProps): JSX.Element {
  return (
    <section>
      <h2 className="text-lg font-bold">{title}</h2>
      <p className="mb-3 text-sm text-gray-500">{note}</p>
      <div className="flex flex-wrap items-center gap-4">{children}</div>
    </section>
  )
}
