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

// Nothing in this app renders `Tooltip.Trigger` — `open` is always driven
// externally by useTooltipGesture — so Base UI never registers a reference
// element to exempt from its outside-press check. That makes it treat a
// press on the anchor itself (the trigger that just opened the tooltip, or a
// touch tap's synthetic post-touchend mousedown/click landing on the same
// element) as an outside press, closing the tooltip a frame after it opened.
function containsEventTarget(
  anchor: TooltipAnchor,
  target: EventTarget | null,
) {
  if (!(target instanceof Node)) return false
  if (anchor instanceof Node) return anchor.contains(target)
  if (anchor && typeof anchor === 'object' && 'current' in anchor)
    return anchor.current instanceof Node
      ? anchor.current.contains(target)
      : false
  return false
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
        if (!next && eventDetails.reason === 'outside-press') {
          if (containsEventTarget(anchor, eventDetails.event.target)) return
          onDismiss(eventDetails.event)
        }
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
