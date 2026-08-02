import type { DictionaryEntry, DictionaryReading } from '@build/cedict'

export interface HanziDefinitionProps {
  entry: DictionaryEntry
  /** Cap on definitions shown across all readings. */
  maxDefinitions?: number
}

/**
 * A CC-CEDICT gloss card. Characters with more than one pronunciation get their
 * senses grouped under it, since which reading is meant is exactly what
 * distinguishes them — 中 as `zhōng` "within" versus `zhòng` "to hit".
 *
 * `maxDefinitions` is a budget across the whole card, spent reading by reading,
 * so a rarer reading can't crowd out the everyday one.
 */
export default function HanziDefinition({
  entry,
  maxDefinitions = Infinity,
}: HanziDefinitionProps): JSX.Element {
  const { hanzi, readings, isProperNoun } = entry
  const [primary, ...rest] = readings

  let budget = maxDefinitions
  const shown = readings.map((reading) => {
    const definitions = reading.definitions.slice(0, Math.max(budget, 0))
    budget -= definitions.length
    return { reading, definitions }
  })
  const truncated = shown.some(
    ({ reading, definitions }) =>
      definitions.length < reading.definitions.length,
  )

  return (
    <div className="w-52 rounded-lg bg-white p-2.5 text-xs shadow-xl ring-1 ring-gray-200">
      <p className="mb-1 align-baseline font-semibold text-gray-900">
        <span className="mr-0.5 text-lg">{hanzi}</span>
        {rest.length === 0 && <Pronunciation reading={primary} />}
        {isProperNoun && (
          <span className="font-normal text-gray-500"> (name)</span>
        )}
      </p>

      {shown.map(({ reading, definitions }) =>
        definitions.length === 0 ? null : (
          <div key={reading.pinyin} className="mb-1 last:mb-0">
            {rest.length > 0 && (
              <p className="font-semibold text-gray-900">
                <Pronunciation reading={reading} />
              </p>
            )}
            <ol className="list-inside list-decimal text-gray-500">
              {definitions.map((def, i) => (
                <li key={`${def}_${i}`}>{def}</li>
              ))}
            </ol>
          </div>
        ),
      )}

      {truncated && <p className="text-gray-500">…</p>}
    </div>
  )
}

function Pronunciation({
  reading,
}: {
  reading: DictionaryReading
}): JSX.Element {
  return (
    <>
      {reading.pinyin}
      {reading.altPronunciation && (
        <span className="font-normal">
          {' '}
          (or {reading.altPronunciation.join(', ')})
        </span>
      )}
    </>
  )
}
