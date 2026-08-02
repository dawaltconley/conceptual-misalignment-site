import type { ComponentProps, ReactNode } from 'react'
import { Tooltip as BaseTooltip } from '@base-ui/react/tooltip'

type TooltipAnchor = ComponentProps<typeof BaseTooltip.Positioner>['anchor']
type TooltipSide = ComponentProps<typeof BaseTooltip.Positioner>['side']

export interface TooltipProps {
  open: boolean
  onDismiss: (event: Event) => void
  anchor: TooltipAnchor
  side?: TooltipSide
  sideOffset?: number
  children: ReactNode
}

/**
 * Base UI wiring only — no gesture/content knowledge. `anchor` accepts a real
 * element, a ref, or a Floating-UI virtual element, so this works for both a
 * real DOM trigger and a synthesized (e.g. canvas hit-test) position.
 */
export default function Tooltip({
  open,
  onDismiss,
  anchor,
  side = 'top',
  sideOffset = 8,
  children,
}: TooltipProps): JSX.Element {
  return (
    <BaseTooltip.Root
      open={open}
      onOpenChange={(next, eventDetails) => {
        if (!next && eventDetails.reason === 'outside-press')
          onDismiss(eventDetails.event)
      }}
    >
      <BaseTooltip.Portal>
        <BaseTooltip.Positioner
          anchor={anchor}
          side={side}
          sideOffset={sideOffset}
          collisionPadding={8}
        >
          <BaseTooltip.Popup className="pointer-events-none">
            {children}
          </BaseTooltip.Popup>
        </BaseTooltip.Positioner>
      </BaseTooltip.Portal>
    </BaseTooltip.Root>
  )
}
