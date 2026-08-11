import type { Source } from '@lib/terms'
import { getHeading, type Heading } from '@lib/headings'
import { toUrl } from '@lib/utils'
import Icon from './Icon'
import { faArrowUpRightFromSquare } from '@fortawesome/pro-regular-svg-icons/faArrowUpRightFromSquare'

interface SourceCardProps {
  source: Source
  headingLevel?: Heading
}

export default function SourceCard({
  source,
  headingLevel,
}: SourceCardProps): JSX.Element {
  const H = headingLevel ? getHeading(headingLevel) : 'p'
  const url = toUrl(source.url)
  const occurrences = source.occurrences
  return (
    <div className="popup max-w-64 space-y-2 p-2 leading-tight shadow-none">
      <H className="font-bold">{source.title}</H>
      <p className="hyphens-auto">{source.description}</p>
      <div className="flex flex-row justify-between border-t border-gray-200 pt-2 text-sm">
        {occurrences && (
          <p>
            <span className="font-semibold">{occurrences}</span> occurrences
          </p>
        )}
        {url && (
          <p>
            <a href={url.href} className="link no-underline">
              <span className="underline">View</span>{' '}
              <Icon icon={faArrowUpRightFromSquare} />
            </a>
          </p>
        )}
      </div>
    </div>
  )
}
