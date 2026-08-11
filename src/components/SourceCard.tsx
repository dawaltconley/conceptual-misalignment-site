import type { Source } from '@lib/terms'
import { getHeading, type Heading } from '@lib/headings'
import { toUrl } from '@lib/utils'
import Icon from './Icon'
import { faArrowUpRightFromSquare } from '@fortawesome/pro-regular-svg-icons/faArrowUpRightFromSquare'
import clsx from 'clsx'

interface SourceCardProps {
  source: Source
  headingLevel?: Heading
  isHighlighted?: boolean
}

export default function SourceCard({
  source,
  headingLevel,
  isHighlighted,
}: SourceCardProps): JSX.Element {
  const H = headingLevel ? getHeading(headingLevel) : 'p'
  const url = toUrl(source.url)
  const occurrences = source.occurrences
  return (
    <div
      className={clsx(
        'popup flex h-full w-64 flex-col gap-y-2 p-2 text-left text-sm leading-tight text-gray-700 shadow-none',
        isHighlighted && 'border-black ring-4 ring-primary',
      )}
    >
      <H className="text-base font-bold leading-none text-gray-900">
        {source.title}
      </H>
      <p className="line-clamp-3 hyphens-auto xl:line-clamp-2">
        {source.description}
      </p>
      <div className="mt-auto flex flex-row justify-between gap-x-4 border-t border-gray-200 pt-2 text-xs">
        {typeof occurrences === 'number' && (
          <p>
            <span className="font-bold text-gray-900">{occurrences}</span>{' '}
            occurrences
          </p>
        )}
        {url && (
          <p>
            <a
              href={url.href}
              target="_blank"
              className={clsx(
                'link no-underline',
                !isHighlighted && 'invisible',
              )}
            >
              <span className="underline">View</span>{' '}
              <Icon icon={faArrowUpRightFromSquare} />
            </a>
          </p>
        )}
      </div>
    </div>
  )
}
