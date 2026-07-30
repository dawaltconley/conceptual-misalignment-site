import fs from 'node:fs'
import path from 'node:path'
import readline from 'node:readline'
import toPinyinTones from 'pinyin-tone'
import { TermSchema } from '@lib/networkx'
import { isNotEmpty } from '@lib/utils'

export interface DictionaryEntry {
  hanzi: string
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

    result[traditional] = {
      hanzi: traditional,
      pinyin: toPinyinTones(pinyinRaw),
      definitions: defs,
    }

    const altPronIndex = defs.findLastIndex((def) =>
      def.startsWith('also pr. '),
    )
    if (altPronIndex < 0) return // no alternate pronunciation, skip

    const altPronRaw = defs.splice(altPronIndex, 1)[0]
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

export async function buildDictionaryFromNetworks(
  ...data: unknown[]
): Promise<Dictionary> {
  const hanzi = data
    .map((d) => TermSchema.parse(d).sources)
    .flat()
    .map((s) => s.cooccurrence?.nodes)
    .filter(isNotEmpty)
    .flat()
    .map((n) => n.id.toString())
  return buildDictionary(hanzi)
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
