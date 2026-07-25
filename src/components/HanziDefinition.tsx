import type { DictionaryEntry } from '@build/cedict'

export interface HanziDefinitionProps {
  entry: DictionaryEntry
  maxDefinitions?: number
}

export default function HanziDefinition({
  entry,
  maxDefinitions = Infinity,
}: HanziDefinitionProps): JSX.Element {
  const { hanzi, pinyin, definitions, altPronunciation } = entry
  return (
    <div className="w-52 rounded-lg bg-white p-2.5 text-xs shadow-xl ring-1 ring-gray-200">
      <p className="mb-1 align-baseline font-semibold text-gray-900">
        <span className="mr-0.5 text-lg">{hanzi}</span> {pinyin}
        {altPronunciation && (
          <span className="font-normal">
            {' '}
            (or {entry.altPronunciation?.join(', ')})
          </span>
        )}
      </p>
      <ol className="list-inside list-decimal text-gray-500">
        {definitions
          .slice(0, maxDefinitions)
          .filter((def) => !def.startsWith('CL:'))
          .map((def, i) => (
            <li key={`${def}_${i}`}>{def}</li>
          ))}
        {definitions.length > maxDefinitions && <li role="presentation">…</li>}
      </ol>
    </div>
  )
}
