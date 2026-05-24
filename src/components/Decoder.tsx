import { useStore } from '../state/store';
import { getDecoderTimesteps } from '../data/index';
import DecoderLayer from './DecoderLayer';
import ContextProjection from './ContextProjection';
import SoftmaxHead from './SoftmaxHead';

export default function Decoder() {
  const { timestep, arquitectura, atencion } = useStore();
  const decTimesteps = getDecoderTimesteps(arquitectura, atencion);

  const visibleCount = Math.min(Math.max(timestep - 7, 0), 6);
  const activeT = timestep >= 8 && timestep <= 13 ? timestep - 7 : null;

  return (
    <div className="flex flex-col gap-3">
      {/* W_c proyección: h_T^(2) → h_0^(dec,1) */}
      <ContextProjection />

      {/* Capas del decoder desdobladas en el tiempo */}
      <div className="flex flex-col gap-1 overflow-x-auto pb-2">
        <DecoderLayer
          layerNum={2}
          decTimesteps={decTimesteps}
          visibleCount={visibleCount}
          activeT={activeT}
        />

        {/* Flechas verticales entre capas */}
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

        <DecoderLayer
          layerNum={1}
          decTimesteps={decTimesteps}
          visibleCount={visibleCount}
          activeT={activeT}
        />

        {timestep === 7 && visibleCount === 0 && (
          <div className="ml-16 text-gray-600 text-xs italic font-mono mt-1">
            Presiona → para comenzar el decoder
          </div>
        )}
      </div>

      {/* Proyección sobre vocabulario */}
      {visibleCount > 0 && (
        <div>
          <h3 className="text-xs text-orange-400/60 font-mono uppercase tracking-wider mb-2">
            Proyección sobre vocabulario
          </h3>
          <SoftmaxHead />
        </div>
      )}
    </div>
  );
}
