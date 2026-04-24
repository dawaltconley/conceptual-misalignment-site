import fs from 'node:fs'
import path from 'node:path'
import readline from 'node:readline'
import toPinyinTones from 'pinyin-tone'

export interface DictionaryEntry {
  pinyin: string
  definitions: string[]
  altPronunciation?: string[]
}

export type Dictionary = Record<string, DictionaryEntry>

const CEDICT_PATH = path.resolve('src/data/cedict_1_0_ts_utf-8_mdbg.txt')
const LINE_RE = /^(\S+)\s+\S+\s+\[([^\]]+)\]\s+\/(.+)\/$/

export async function buildDictionary(
  chars: Iterable<string>,
): Promise<Dictionary> {
  const targets = new Set(chars)
  const result: Dictionary = {}

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
    // If we already have an entry, keep it unless this one has lowercase pinyin
    // (lowercase = common entry; uppercase = proper noun / surname)
    if (result[traditional] && /^[A-Z]/.test(pinyinRaw)) return

    const defs = defsRaw
      .split('/')
      .filter((def) => Boolean(def) && !def.startsWith('CL:'))
    const altPronIndex = defs.findLastIndex((def) =>
      def.startsWith('also pr. '),
    )
    const altPronRaw = defs.splice(altPronIndex, 1)[0]

    result[traditional] = {
      pinyin: toPinyinTones(pinyinRaw),
      definitions: defs,
    }

    if (altPronRaw[9] === '[' && altPronRaw[altPronRaw.length - 1] === ']') {
      const altPron = altPronRaw
        .slice(10, -1)
        .split(' ')
        .filter(Boolean)
        .map(toPinyinTones)
      if (altPron.length > 0) {
        result[traditional].altPronunciation = altPron
      }
    }
  })

  return new Promise((resolve, reject) => {
    rl.on('close', () => resolve(result))
    rl.on('error', (e) => reject(e))
  })
}
