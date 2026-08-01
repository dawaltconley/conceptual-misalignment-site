import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

/**
 * Class merger for the shadcn components under `src/components/ui` and the
 * wrappers around them: `clsx` conditionals, then `tailwind-merge` to let a
 * caller's utility class beat a conflicting one baked into the component.
 * Hand-written components elsewhere use plain `clsx` + component classes.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs))
}
