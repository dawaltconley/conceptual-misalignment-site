import fs from 'node:fs'
import path from 'node:path'
import readline from 'node:readline'
import toPinyinTones from 'pinyin-tone'

/** One pronunciation of a headword, with the senses that belong to it. */
export interface DictionaryReading {
  /** Tone-marked pinyin for this reading, e.g. `zhòng`. */
  pinyin: string
  definitions: string[]
  /** The `also pr.` variants CC-CEDICT notes for this reading. */
  altPronunciation?: string[]
  /**
   * CC-CEDICT capitalises the pinyin of proper-noun senses (`Zhong1` = the
   * surname). These sort last and are worth labelling as such.
   */
  isProperNoun: boolean
}

export interface DictionaryEntry {
  hanzi: string
  /** Every reading, ordinary senses before proper nouns. Never empty. */
  readings: DictionaryReading[]
  /** The first reading's pinyin — what a one-line label shows. */
  pinyin: string
}

export type Dictionary = Record<string, DictionaryEntry>

const CEDICT_PATH = path.resolve('src/data/cedict_1_0_ts_utf-8_mdbg.txt')
const LINE_RE = /^(\S+)\s+\S+\s+\[([^\]]+)\]\s+\/(.+)\/$/

/**
 * Tone marks, preserving CC-CEDICT's capitalisation of proper nouns —
 * `pinyin-tone` only marks lowercase input, so `Zhong1` would come back
 * unchanged rather than as `Zhōng`.
 */
function toTonePinyin(raw: string): string {
  const marked = toPinyinTones(raw.toLowerCase())
  if (!/^[A-Z]/.test(raw)) return marked
  return marked.charAt(0).toUpperCase() + marked.slice(1)
}

/**
 * Senses that only point elsewhere — `used in 惡心[e3 xin1]`, `variant of …`.
 * A reading with nothing but these is a cross-reference, not a meaning, and
 * shouldn't be what a character's label shows.
 */
const CROSS_REFERENCE = /^(used in|see |variant of|old variant of|abbr\. for)/i

const isCrossReference = (reading: DictionaryReading): boolean =>
  reading.definitions.every((def) => CROSS_REFERENCE.test(def))

/**
 * Sort key: real senses first, then proper nouns, then cross-references. Within
 * a rank CC-CEDICT's own order is kept. Without this, 樂 leads with `lào`
 * ("used in place names") and 惡 with `ě`, purely because those lines come
 * first in the file.
 */
const readingRank = (reading: DictionaryReading): number =>
  (isCrossReference(reading) ? 2 : 0) + (reading.isProperNoun ? 1 : 0)

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

/**
 * Build a dictionary for `chars` from the bundled CC-CEDICT.
 *
 * A character usually has several CC-CEDICT lines — one per pronunciation, plus
 * a capitalised one for any surname or place-name sense. They are merged into a
 * single entry keyed by the headword, keeping each pronunciation's senses
 * separate: 中 is `zhōng` (within, middle), `zhòng` (to hit a target) and
 * `Zhōng` (China; surname), not whichever line happened to come last.
 *
 * Lines sharing a pronunciation are folded together, and readings are ordered
 * with proper nouns last so `entry.pinyin` is the everyday reading.
 */
export async function buildDictionary(
  chars: Iterable<string>,
): Promise<Dictionary> {
  const targets = new Set(chars)
  // Readings in file order, keyed headword → pinyin, so repeated pronunciations
  // (CC-CEDICT lists some once per simplified form) accumulate rather than clash.
  const collected = new Map<string, Map<string, DictionaryReading>>()

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

    let readings = collected.get(traditional)
    if (!readings) {
      readings = new Map()
      collected.set(traditional, readings)
    }

    const pinyin = toTonePinyin(pinyinRaw)
    const existing = readings.get(pinyin)
    if (existing) {
      // Same pronunciation seen again: union the senses, keeping first order.
      const seen = new Set(existing.definitions)
      existing.definitions.push(...definitions.filter((d) => !seen.has(d)))
      existing.altPronunciation ??= altPronunciation
      return
    }

    readings.set(pinyin, {
      pinyin,
      definitions,
      altPronunciation,
      isProperNoun: /^[A-Z]/.test(pinyinRaw),
    })
  })

  return new Promise((resolve, reject) => {
    rl.on('close', () => {
      const result: Dictionary = {}
      for (const [hanzi, readings] of collected) {
        // Stable sort, so within a rank CC-CEDICT's own order survives.
        const ordered = [...readings.values()].sort(
          (a, b) => readingRank(a) - readingRank(b),
        )
        result[hanzi] = { hanzi, readings: ordered, pinyin: ordered[0].pinyin }
      }
      resolve(result)
    })
    rl.on('error', (e) => reject(e))
  })
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
