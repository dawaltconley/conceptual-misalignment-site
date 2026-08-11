import type { EmbeddingNode } from '@lib/embeddings'
import type { Dictionary, DictionaryEntry } from '@lib/build/cedict'
import { useRef } from 'react'
import { Dialog } from '@base-ui/react/dialog'
import clsx from 'clsx'

export interface CommunityDialogProps {
  title: string
  description: string
  nodes: EmbeddingNode[]
  handle?: Dialog.Handle<{ id: string }>
  dictionary?: Dictionary
}

export default function CommunityDialog({
  title,
  description,
  nodes,
  handle,
  dictionary,
}: CommunityDialogProps) {
  const popupRef = useRef<HTMLDivElement>(null)
  return (
    <Dialog.Root handle={handle}>
      {!handle && (
        <Dialog.Trigger className="btn">View community</Dialog.Trigger>
      )}
      <Dialog.Portal>
        <Dialog.Backdrop className="dialog__backdrop" />
        <Dialog.Viewport className="dialog__viewport">
          <Dialog.Popup
            ref={popupRef}
            initialFocus={popupRef}
            className="dialog__popup"
          >
            <Dialog.Title className="mb-4 text-xl font-bold">
              {title}
            </Dialog.Title>
            <Dialog.Description>{description}</Dialog.Description>
            <table className="my-4 w-full border-collapse">
              <thead>
                <tr className="*:border *:border-gray-300 *:p-1">
                  <th scope="col">Term</th>
                  {dictionary && (
                    <>
                      <th scope="col">Pinyin</th>
                      <th scope="col">Definition</th>
                    </>
                  )}
                  <th scope="col">Strength</th>
                  <th scope="col">Documents</th>
                  <th scope="col">Eigenvector</th>
                </tr>
              </thead>
              <tbody>
                {nodes.map((n) => {
                  const entry: DictionaryEntry | undefined = dictionary?.[n.id]
                  const readings = entry?.readings.map<[string, string]>(
                    (e) => [e.pinyin, e.definitions.join(', ')],
                  ) || [null]
                  return readings.map((r, i) => (
                    <tr
                      key={n.id + (r ? r[0] : '')}
                      className={clsx(
                        '*:border *:border-gray-300 *:p-1',
                        n.target && 'font-bold',
                      )}
                    >
                      {i === 0 && (
                        <th
                          scope="row"
                          rowSpan={readings.length}
                          className={clsx(
                            'text-left',
                            !n.target && 'font-normal',
                          )}
                        >
                          {n.id}
                        </th>
                      )}
                      {r && (
                        <>
                          <td>{r[0]}</td>
                          <td className="max-w-prose">{r[1]}</td>
                        </>
                      )}
                      {i === 0 && (
                        <>
                          <td rowSpan={readings.length} className="text-right">
                            {n.strength.toPrecision(5)}
                          </td>
                          <td rowSpan={readings.length} className="text-right">
                            {n.doc_freq.toFixed(0)}
                          </td>
                          <td rowSpan={readings.length} className="text-right">
                            {n.eigenvector.toPrecision(5)}
                          </td>
                        </>
                      )}
                    </tr>
                  ))
                })}
              </tbody>
            </table>
            <div className="mt-2">
              <Dialog.Close className="btn">Close</Dialog.Close>
            </div>
          </Dialog.Popup>
        </Dialog.Viewport>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
