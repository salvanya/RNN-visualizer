import { useStore } from "../state/store";

type Chip = {
  symbol: string;
  value: number;
  label: string;
  color: string;
};

export default function DimensionsLegend() {
  const { modo } = useStore();
  const isTranslation = modo === "translation";

  const embeddingChip: Chip = {
    symbol: "d",
    value: 4,
    label: "embedding",
    color: "#22d3ee", // cyan-400
  };

  const encoderChips: Chip[] = [
    { symbol: "m", value: 3, label: "encoder · capa 1", color: "#60a5fa" }, // blue-400
    { symbol: "l", value: 4, label: "encoder · capa 2", color: "#60a5fa" },
  ];

  const decoderChips: Chip[] = [
    { symbol: "p", value: 5, label: "decoder · capa 1", color: "#fb923c" }, // orange-400
    { symbol: "q", value: 3, label: "decoder · capa 2", color: "#fb923c" },
  ];

  const vocabChip: Chip = {
    symbol: "V",
    value: 12,
    label: "vocab. decoder",
    color: "#a78bfa", // violet-400
  };

  const chips: Chip[] = [
    embeddingChip,
    ...encoderChips,
    ...(isTranslation ? decoderChips : []),
    ...(isTranslation ? [vocabChip] : []),
  ];

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 px-6 py-2 bg-gray-900/60 border-b border-gray-800 text-[11px]">
      <span
        className="text-[10px] uppercase tracking-[0.14em] text-gray-500"
        style={{ fontFamily: '"Space Grotesk", Inter, sans-serif' }}
      >
        Dimensiones
      </span>

      {chips.map((chip, i) => (
        <div key={chip.symbol} className="flex items-center gap-2">
          {i > 0 && <span className="text-gray-700">·</span>}
          <div className="flex items-baseline gap-1.5">
            <span
              className="font-mono font-semibold"
              style={{ color: chip.color, fontFamily: '"JetBrains Mono", monospace' }}
            >
              {chip.symbol}={chip.value}
            </span>
            <span className="text-gray-500">{chip.label}</span>
          </div>
        </div>
      ))}

      <span className="text-gray-700">·</span>
      <span className="text-gray-500">
        encoder: <span className="text-gray-400">2 capas</span>
        {isTranslation && (
          <>
            {" "}· decoder: <span className="text-gray-400">2 capas</span>
          </>
        )}
      </span>
    </div>
  );
}
