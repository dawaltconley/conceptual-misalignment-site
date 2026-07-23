import type { Term } from '@lib/networkx'
import type { Dictionary } from '@build/cedict'
import MultiNetwork from './MultiNetwork'
import { useState } from 'react'
import clsx from 'clsx'

interface TermNetworkProps {
  terms: Term[]
  sourceAlign?: 'left' | 'right'
  dictionary?: Dictionary
}

export default function TermNetwork({
  terms,
  sourceAlign = 'right',
  dictionary,
}: TermNetworkProps): JSX.Element {
  const [selected, setSelected] = useState(0)

  return (
    <div>
      {terms.length > 1 && (
        <div
          className={clsx(
            'mb-3 flex gap-2',
            sourceAlign === 'left' ? 'justify-start' : 'justify-end',
          )}
        >
          {terms.map((term, i) => (
            <button
              key={term.term}
              className={clsx(
                'rounded border border-gray-900 px-3 py-1 text-sm capitalize duration-150 hover:bg-red-200',
                i === selected && 'border-red-500 bg-red-500 text-white',
              )}
              onClick={() => setSelected(i)}
            >
              {term.term}
            </button>
          ))}
        </div>
      )}
      <MultiNetwork
        data={terms[selected]}
        sourceAlign={sourceAlign}
        dictionary={dictionary}
      />
    </div>
  )
}
