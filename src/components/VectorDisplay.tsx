import { fmt } from '../utils/math-format';

interface Props {
  values: number[];
  label?: string;
  color?: string;
  withBars?: boolean;
}

export default function VectorDisplay({ values, label, color = '#94a3b8', withBars }: Props) {
  return (
    <div>
      {label && (
        <div className="text-[10px] text-gray-500 font-mono mb-0.5">{label}</div>
      )}
      <div className="flex flex-col gap-0.5">
        {values.map((v, i) => (
          <div key={i} className="flex items-center gap-1.5">
            {withBars && (
              <div className="w-10 flex items-center justify-end">
                {v < 0 && (
                  <div
                    className="h-2 rounded-sm"
                    style={{ width: `${Math.min(Math.abs(v), 1) * 36}px`, backgroundColor: '#f87171' }}
                  />
                )}
                {v >= 0 && <div className="w-full" />}
              </div>
            )}
            {withBars && (
              <div className="w-10 flex items-center">
                {v >= 0 && (
                  <div
                    className="h-2 rounded-sm"
                    style={{ width: `${Math.min(Math.abs(v), 1) * 36}px`, backgroundColor: '#4ade80' }}
                  />
                )}
              </div>
            )}
            <span className="font-mono text-[11px]" style={{ color }}>
              {fmt(v)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
