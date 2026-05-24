import { useStore } from '../state/store';
import { getEncoderTimesteps } from '../data/index';
import Layer from './Layer';

export default function Encoder() {
  const { timestep, arquitectura, modo } = useStore();

  const encTimesteps = getEncoderTimesteps(arquitectura, modo);

  // Cells visible: 0 at t=0, grow to 7 by t=7, stay at 7 for decoder phase
  const visibleCount = Math.min(Math.max(timestep, 0), 7);
  // Which cell is "just computed" (highlight): only during encoder phase
  const activeT = timestep >= 1 && timestep <= 7 ? timestep : null;

  return (
    <div className="flex flex-col gap-1 overflow-x-auto pb-2">
      {/* Layer 2 (top — receives h_t^(1) as input) */}
      <Layer
        layerNum={2}
        encTimesteps={encTimesteps}
        visibleCount={visibleCount}
        activeT={activeT}
      />

      {/* Vertical connection hint between layers */}
      <div className="flex items-center">
        <div className="w-16 shrink-0" />
        <div className="flex items-center gap-px">
          {visibleCount > 0 &&
            Array.from({ length: visibleCount }, (_, i) => (
              <div
                key={i}
                className="flex justify-center"
                style={{ width: i === 0 ? '96px' : '116px', paddingLeft: i === 0 ? 0 : '20px' }}
              >
                <div className="text-gray-700 text-[10px] font-mono select-none">↑</div>
              </div>
            ))}
        </div>
      </div>

      {/* Layer 1 (bottom — receives x_t embeddings) */}
      <Layer
        layerNum={1}
        encTimesteps={encTimesteps}
        visibleCount={visibleCount}
        activeT={activeT}
      />

      {/* Hint when no cells are visible yet */}
      {visibleCount === 0 && (
        <div className="ml-16 text-gray-600 text-xs italic font-mono mt-1">
          Presiona → o reproduce para comenzar
        </div>
      )}
    </div>
  );
}
