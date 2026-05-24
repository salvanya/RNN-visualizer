interface Props {
  matrix: number[][];
  label?: string;
  rowLabels?: string[];
  colLabels?: string[];
  maxAbs?: number;
}

function valToColor(v: number, maxAbs: number): string {
  if (maxAbs === 0) return 'rgb(30,30,30)';
  const t = Math.max(-1, Math.min(1, v / maxAbs));
  if (t >= 0) {
    // white → green
    const r = Math.round(255 - t * 200);
    const g = Math.round(255 - t * 55);
    const b = Math.round(255 - t * 200);
    return `rgb(${r},${g},${b})`;
  } else {
    // white → red
    const abs = -t;
    const r = Math.round(255 - abs * 55);
    const g = Math.round(255 - abs * 200);
    const b = Math.round(255 - abs * 200);
    return `rgb(${r},${g},${b})`;
  }
}

export default function MatrixDisplay({ matrix, label, rowLabels, colLabels, maxAbs }: Props) {
  const computed = maxAbs ?? Math.max(...matrix.flat().map(Math.abs), 0.01);

  return (
    <div className="flex flex-col gap-1">
      {label && (
        <span className="font-mono text-[10px] text-gray-400">{label}</span>
      )}
      <div className="overflow-auto">
        <table className="border-collapse text-[8px] font-mono">
          {colLabels && (
            <thead>
              <tr>
                {rowLabels && <th />}
                {colLabels.map((cl, j) => (
                  <th key={j} className="text-gray-500 pb-0.5 px-0.5 text-center font-normal">
                    {cl}
                  </th>
                ))}
              </tr>
            </thead>
          )}
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i}>
                {rowLabels && (
                  <td className="text-gray-500 pr-1 text-right font-normal">{rowLabels[i]}</td>
                )}
                {row.map((v, j) => (
                  <td
                    key={j}
                    title={v.toFixed(3)}
                    className="w-6 h-6 text-center rounded-sm"
                    style={{
                      backgroundColor: valToColor(v, computed),
                      color: Math.abs(v / computed) > 0.5 ? 'rgba(0,0,0,0.75)' : 'rgba(0,0,0,0.5)',
                    }}
                  >
                    {v.toFixed(1)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
