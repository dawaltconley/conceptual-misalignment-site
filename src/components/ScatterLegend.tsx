export interface LegendLabel {
  id: string
  color: string
  description: string
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
      {labels.map(({ id, color, description }) => (
        <div
          key={id}
          className="flex flex-row items-center gap-1"
          onMouseOver={() => onHover && onHover(id)}
          onMouseOut={() => onHover && onHover(null)}
          onClick={() => onClick && onClick(id)}
        >
          <span
            className="relative top-px inline-block aspect-square h-2 shrink overflow-ellipsis border border-black"
            style={{ backgroundColor: color }}
          ></span>
          <span className="w-full overflow-x-hidden text-ellipsis whitespace-nowrap">
            {description}
          </span>
        </div>
      ))}
    </div>
  )
}

// export default function ScatterLegend({
//   labels,
//   onHover,
//   onClick,
// }: ScatterLegendProps): JSX.Element {
//   return (
//     <table className="border-spacing-x-2 border-spacing-y-1 text-sm">
//       <tbody>
//         {labels.map(({ id, color, description }) => (
//           <tr key={id}>
//             <th scope="row">
//               <span
//                 className="inline-block aspect-square h-2 overflow-ellipsis border border-black"
//                 style={{ backgroundColor: color }}
//               ></span>
//             </th>
//             <td className="h-[1em] w-full text-ellipsis">{description}</td>
//           </tr>
//         ))}
//       </tbody>
//     </table>
//   )
// }
