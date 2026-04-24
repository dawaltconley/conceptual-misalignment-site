import fs from 'node:fs'
import path from 'node:path'
import readline from 'node:readline'
import toPinyinTones from 'pinyin-tone'

export interface DictionaryEntry {
  pinyin: string
  definitions: string[]
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

  for await (const line of rl) {
    if (line.startsWith('#')) continue
    const m = line.match(LINE_RE)
    if (!m) continue
    const [, traditional, pinyinRaw, defsRaw] = m
    if (!targets.has(traditional)) continue
    // If we already have an entry, keep it unless this one has lowercase pinyin
    // (lowercase = common entry; uppercase = proper noun / surname)
    if (result[traditional] && /^[A-Z]/.test(pinyinRaw)) continue
    result[traditional] = {
      pinyin: toPinyinTones(pinyinRaw),
      definitions: defsRaw.split('/').filter(Boolean),
    }
  }

  return result
}
