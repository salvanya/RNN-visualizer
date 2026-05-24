interface Props {
  showC?: boolean;
}

export default function EdgeLegend({ showC = false }: Props) {
  return (
    <div className="flex items-center gap-4 px-3 py-1.5 rounded-lg bg-gray-900/40 border border-gray-800/60 text-[10px] font-mono text-gray-400 w-fit">
      <span className="text-gray-500 uppercase tracking-wider text-[9px]">Conexiones</span>

      <LegendItem swatch={<EdgeSwatch color="#22c55e" />} label="W &gt; 0" />
      <LegendItem swatch={<EdgeSwatch color="#ef4444" />} label="W &lt; 0" />
      <LegendItem
        swatch={
          <svg width={24} height={6}>
            <line x1={1} y1={3} x2={8} y2={3} stroke="#ef4444" strokeWidth={0.6} />
            <line x1={1} y1={3} x2={11} y2={3} stroke="#22c55e" strokeWidth={1.4} />
            <line x1={13} y1={3} x2={23} y2={3} stroke="#22c55e" strokeWidth={2.4} />
          </svg>
        }
        label="grosor ∝ |W|"
      />
      {showC && (
        <LegendItem
          swatch={
            <svg width={24} height={6}>
              <line
                x1={1}
                y1={3}
                x2={23}
                y2={3}
                stroke="#facc15"
                strokeWidth={1.6}
                strokeDasharray="4 2"
                strokeOpacity={0.8}
              />
            </svg>
          }
          label="C_t (estado de celda LSTM)"
        />
      )}
    </div>
  );
}

function LegendItem({ swatch, label }: { swatch: React.ReactNode; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      {swatch}
      <span>{label}</span>
    </div>
  );
}

function EdgeSwatch({ color }: { color: string }) {
  return (
    <svg width={24} height={6}>
      <line x1={1} y1={3} x2={23} y2={3} stroke={color} strokeWidth={1.8} strokeOpacity={0.85} />
    </svg>
  );
}
