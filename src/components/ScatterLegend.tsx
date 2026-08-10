import { Dialog } from '@base-ui/react/dialog'
import clsx from 'clsx'

export interface LegendLabel {
  id: string
  color: string
  description: string
  dialog?: Dialog.Handle<{ id: string }>
}

export interface ScatterLegendProps {
  labels: LegendLabel[]
  onHover?: (id: string | null) => void
  onClick?: (id: string) => void
}

export default function ScatterLegend({
  labels,
  onHover,
  onClick,
}: ScatterLegendProps): JSX.Element {
  return (
    <div>
      {labels.map(({ id, color, description, dialog }) => {
        const E = dialog ? Dialog.Trigger : 'div'
        return (
          <E
            key={id}
            handle={dialog}
            payload={dialog && { id }}
            className={clsx(
              'flex w-full flex-row items-center gap-1',
              dialog && 'cursor-pointer',
            )}
            onMouseOver={() => onHover && onHover(id)}
            onMouseOut={() => onHover && onHover(null)}
            onClick={() => onClick && onClick(id)}
          >
            <ColorSwatch color={color} />
            <div className="w-full overflow-x-hidden text-ellipsis whitespace-nowrap">
              {description}
            </div>
          </E>
        )
      })}
    </div>
  )
}

interface ColorSwatchProps {
  color: string
}

function ColorSwatch({ color }: ColorSwatchProps) {
  return (
    <span
      className="relative top-px inline-block aspect-square h-2 shrink overflow-ellipsis border border-black"
      style={{ backgroundColor: color }}
    ></span>
  )
}
