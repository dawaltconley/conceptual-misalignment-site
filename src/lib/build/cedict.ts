import fs from 'node:fs'
import path from 'node:path'
import readline from 'node:readline'
import toPinyinTones from 'pinyin-tone'

/** One pronunciation of a headword, with the senses that belong to it. */
export interface DictionaryReading {
  /** Tone-marked pinyin for this reading, e.g. `zhòng`. */
  pinyin: string
  /** Empty only for a headword whose reading was assembled per-character. */
  definitions: string[]
  /** The `also pr.` variants CC-CEDICT notes for this reading. */
  altPronunciation?: string[]
}

export interface DictionaryEntry {
  hanzi: string
  /** Every reading, meanings before cross-references. Never empty. */
  readings: DictionaryReading[]
  /** The first reading's pinyin — what a one-line label shows. */
  pinyin: string
  /**
   * CC-CEDICT knows this headword *only* as a name — 舜, 孟子, 秦. Ordinary
   * senses always win, so this is all-or-nothing, and it's the flag downstream
   * filters on.
   */
  isProperNoun?: boolean
}

export type Dictionary = Record<string, DictionaryEntry>

const CEDICT_PATH = path.resolve('src/data/cedict_1_0_ts_utf-8_mdbg.txt')
const LINE_RE = /^(\S+)\s+\S+\s+\[([^\]]+)\]\s+\/(.+)\/$/

/**
 * Tone marks, preserving CC-CEDICT's capitalisation of names — `pinyin-tone`
 * only marks lowercase input, so `Shun4` would come back unchanged rather than
 * as `Shùn`.
 */
function toTonePinyin(raw: string): string {
  const marked = toPinyinTones(raw.toLowerCase())
  if (!/^[A-Z]/.test(raw)) return marked
  return marked.charAt(0).toUpperCase() + marked.slice(1)
}

/**
 * CC-CEDICT capitalises the pinyin of proper-noun senses — `Zhong1` is the
 * surname 中, `Le4` the place name 樂. They're set aside and only used for
 * headwords that have no ordinary sense at all.
 */
const isProperNoun = (pinyinRaw: string): boolean => /^[A-Z]/.test(pinyinRaw)

/**
 * Senses that only point elsewhere — `used in 惡心[e3 xin1]`, `variant of …`.
 * A reading with nothing but these is a cross-reference, not a meaning, and
 * shouldn't be what a character's label shows.
 */
const CROSS_REFERENCE = /^(used in|see |variant of|old variant of|abbr\. for)/i

const isCrossReference = (reading: DictionaryReading): boolean =>
  reading.definitions.every((def) => CROSS_REFERENCE.test(def))

/**
 * Sort key: real senses before cross-references. Within a rank CC-CEDICT's own
 * order is kept. Without this, 樂 leads with `lào` ("used in place names") and
 * 惡 with `ě`, purely because those lines come first in the file.
 */
const readingRank = (reading: DictionaryReading): number =>
  isCrossReference(reading) ? 1 : 0

/**
 * Pull the trailing `also pr. [xx1 yy2]` note out of a sense list, since it
 * annotates the reading rather than being a definition of its own.
 */
function takeAltPronunciation(definitions: string[]): string[] | undefined {
  const index = definitions.findLastIndex((def) => def.startsWith('also pr. '))
  if (index < 0) return undefined

  const raw = definitions.splice(index, 1)[0]
  if (raw[9] !== '[' || raw[raw.length - 1] !== ']') return undefined

  const alt = raw.slice(10, -1).split(' ').filter(Boolean).map(toTonePinyin)
  return alt.length > 0 ? alt : undefined
}

/** Readings keyed headword → pinyin, in file order. */
type Collected = Map<string, Map<string, DictionaryReading>>

/**
 * One streaming pass over CC-CEDICT, collecting every line whose headword is in
 * `targets`. Ordinary senses and names land in separate maps, since names are
 * only ever a fallback.
 */
async function scanCedict(
  targets: ReadonlySet<string>,
): Promise<{ senses: Collected; names: Collected }> {
  const senses: Collected = new Map()
  const names: Collected = new Map()

  const rl = readline.createInterface({
    input: fs.createReadStream(CEDICT_PATH, { encoding: 'utf8' }),
    crlfDelay: Infinity,
  })

  rl.on('line', (line) => {
    if (line.startsWith('#')) return
    const m = line.match(LINE_RE)
    if (!m) return
    const [, traditional, pinyinRaw, defsRaw] = m
    if (!targets.has(traditional)) return

    const definitions = defsRaw
      .split('/')
      .filter((def) => Boolean(def) && !def.startsWith('CL:'))
    const altPronunciation = takeAltPronunciation(definitions)
    if (!definitions.length) return

    const collected = isProperNoun(pinyinRaw) ? names : senses
    let readings = collected.get(traditional)
    if (!readings) {
      readings = new Map()
      collected.set(traditional, readings)
    }

    const pinyin = toTonePinyin(pinyinRaw)
    const existing = readings.get(pinyin)
    if (existing) {
      // Same pronunciation seen again (CC-CEDICT lists some once per simplified
      // form): union the senses, keeping first order.
      const seen = new Set(existing.definitions)
      existing.definitions.push(...definitions.filter((d) => !seen.has(d)))
      existing.altPronunciation ??= altPronunciation
      return
    }

    readings.set(pinyin, { pinyin, definitions, altPronunciation })
  })

  return new Promise((resolve, reject) => {
    rl.on('close', () => resolve({ senses, names }))
    rl.on('error', (e) => reject(e))
  })
}

/**
 * Fold one headword's collected readings into an entry. Ordinary senses win
 * outright; the name is a fallback, so 中 is "within" rather than the surname
 * but 舜 still gets its sage-king.
 */
function toEntry(
  hanzi: string,
  { senses, names }: { senses: Collected; names: Collected },
): DictionaryEntry | null {
  const readings = senses.get(hanzi) ?? names.get(hanzi)
  if (!readings) return null
  // Stable sort, so within a rank CC-CEDICT's own order survives.
  const ordered = [...readings.values()].sort(
    (a, b) => readingRank(a) - readingRank(b),
  )
  return {
    hanzi,
    readings: ordered,
    pinyin: ordered[0].pinyin,
    ...(senses.has(hanzi) ? {} : { isProperNoun: true }),
  }
}

/**
 * Build a dictionary for `chars` from the bundled CC-CEDICT.
 *
 * A character usually has several CC-CEDICT lines, one per pronunciation. They
 * are merged into a single entry keyed by the headword, keeping each
 * pronunciation's senses separate: 中 is `zhōng` (within, middle) *and* `zhòng`
 * (to hit a target), not whichever line happened to come last.
 *
 * Lines sharing a pronunciation are folded together, and readings are ordered
 * with cross-references last so `entry.pinyin` is the everyday reading. Names
 * are set aside: 中 keeps only `zhōng`/`zhòng`, but a headword CC-CEDICT knows
 * *only* as a name (舜, 孟子) gets that, marked `isProperNoun`.
 *
 * A multi-character headword that isn't in CC-CEDICT at all falls back to its
 * characters' readings joined together — pinyin with no definitions, 許子 as
 * `Xǔ zǐ`. Costs a second pass, taken only when something is missing.
 */
export async function buildDictionary(
  chars: Iterable<string>,
): Promise<Dictionary> {
  const targets = new Set(chars)
  const collected = await scanCedict(targets)

  const result: Dictionary = {}
  for (const hanzi of targets) {
    const entry = toEntry(hanzi, collected)
    if (entry) result[hanzi] = entry
  }

  // Multi-character headwords CC-CEDICT has never heard of — mostly Mengzi's
  // minor figures (許子, 瞽瞍). A per-character reading is still better than a
  // bare glyph, so gather the characters we don't already have and go again.
  const unresolved = [...targets].filter(
    (hanzi) => !result[hanzi] && [...hanzi].length > 1,
  )
  const wanted = new Set(
    unresolved.flatMap((hanzi) => [...hanzi]).filter((char) => !result[char]),
  )
  const parts = wanted.size > 0 ? await scanCedict(wanted) : null

  for (const hanzi of unresolved) {
    const syllables = [...hanzi].map(
      (char) => result[char]?.pinyin ?? (parts && toEntry(char, parts)?.pinyin),
    )
    // All or nothing: half a reading would be misleading rather than useful.
    if (syllables.some((syllable) => !syllable)) continue
    const pinyin = syllables.join(' ')
    result[hanzi] = { hanzi, pinyin, readings: [{ pinyin, definitions: [] }] }
  }

  return result
}

/**
 * Build a dictionary over every node id in a set of per-source NetworkData files,
 * given their web paths (e.g. from the master term index). Read at build time
 * from `publicDir`; missing/unreadable files are skipped.
 */
export async function buildDictionaryFromFiles(
  webPaths: Iterable<string>,
  publicDir = 'public',
): Promise<Dictionary> {
  const hanzi: string[] = []
  for (const webPath of webPaths) {
    const file = path.join(publicDir, webPath)
    if (!fs.existsSync(file)) continue
    try {
      const data = JSON.parse(fs.readFileSync(file, 'utf8'))
      for (const node of data?.network?.nodes ?? []) {
        hanzi.push(String(node.id))
      }
    } catch {
      // skip malformed file
    }
  }
  return buildDictionary(hanzi)
}
