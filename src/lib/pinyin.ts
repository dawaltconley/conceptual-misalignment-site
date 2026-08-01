/**
 * Search keys for pinyin, so a Chinese term can be found from a plain
 * alphanumeric keyboard: 仁 matches `ren2`, `ren`, or `rén`.
 *
 * These are string transforms only — nothing here converts hanzi to pinyin.
 * A reading is supplied by the caller (`CC-CEDICT` via `@build/cedict` already
 * has one for every character in the corpus) and expanded into the forms
 * someone might type.
 */

/** Tone-marked vowel → the plain letter and its tone number. */
// prettier-ignore
const TONE_MARKS: Record<string, [letter: string, tone: number]> = {
  ā: ['a', 1], á: ['a', 2], ǎ: ['a', 3], à: ['a', 4],
  ē: ['e', 1], é: ['e', 2], ě: ['e', 3], è: ['e', 4],
  ī: ['i', 1], í: ['i', 2], ǐ: ['i', 3], ì: ['i', 4],
  ō: ['o', 1], ó: ['o', 2], ǒ: ['o', 3], ò: ['o', 4],
  ū: ['u', 1], ú: ['u', 2], ǔ: ['u', 3], ù: ['u', 4],
  ǖ: ['v', 1], ǘ: ['v', 2], ǚ: ['v', 3], ǜ: ['v', 4],
  ü: ['v', 0],
}

/** Syllable boundaries as CC-CEDICT and common input both write them. */
const SYLLABLE_SEPARATORS = /[\s'·-]+/

/**
 * One syllable's tone marks resolved to a trailing digit: `rén` → `ren2`,
 * `de` → `de5` (neutral). A syllable that already ends in a digit is left
 * alone, so numbered input passes through unchanged.
 */
export function toNumbered(syllable: string): string {
  const lower = syllable.toLowerCase()
  if (/\d$/.test(lower)) return lower

  let tone = 5 // neutral unless a mark says otherwise
  let letters = ''
  for (const char of lower) {
    const mark = TONE_MARKS[char]
    if (!mark) {
      letters += char
      continue
    }
    const [letter, marked] = mark
    letters += letter
    if (marked > 0) tone = marked
  }
  return letters + tone
}

/** The same syllable with no tone at all: `rén` → `ren`. */
export function toToneless(syllable: string): string {
  return toNumbered(syllable).replace(/\d$/, '')
}

/**
 * Every form of a reading someone might type, for use as a combobox option's
 * `keywords`. For `rén yì`: the reading as given, `ren2`/`yi4` and their
 * joined form, the toneless `ren`/`yi` and *their* joined form, and — for
 * multi-syllable readings — the initials `ry`.
 *
 * `ü` is keyed as both `v` and `u`, since both get typed.
 */
export function pinyinKeywords(pinyin: string): string[] {
  const syllables = pinyin.trim().split(SYLLABLE_SEPARATORS).filter(Boolean)
  if (!syllables.length) return []

  const numbered = syllables.map(toNumbered)
  const toneless = numbered.map((s) => s.replace(/\d$/, ''))

  const keys = [
    pinyin.trim().toLowerCase(),
    ...numbered,
    ...toneless,
    numbered.join(''),
    toneless.join(''),
  ]
  if (syllables.length > 1) keys.push(toneless.map((s) => s[0]).join(''))

  // `lv`/`lu` are both plausible spellings of `lǜ`.
  for (const key of [...keys]) {
    if (key.includes('v')) keys.push(key.replaceAll('v', 'u'))
  }

  return [...new Set(keys)].filter(Boolean)
}
