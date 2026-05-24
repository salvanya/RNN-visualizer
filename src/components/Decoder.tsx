import { useStore } from '../state/store';
import { getDecoderTimesteps } from '../data/index';
import DecoderLayer from './DecoderLayer';
import ContextProjection from './ContextProjection';
import SoftmaxHead from './SoftmaxHead';
import AttentionLayer from './AttentionLayer';
import { SectionHeader, WaitingMessage } from './Section';

export default function Decoder() {
  const { timestep, arquitectura, atencion } = useStore();
  const decTimesteps = getDecoderTimesteps(arquitectura, atencion);

  const visibleCount = Math.min(Math.max(timestep - 7, 0), 6);
  const activeT = timestep >= 8 && timestep <= 13 ? timestep - 7 : null;

  return (
    <div className="flex flex-col gap-5">
      {/* W_c proyección: h_T^(2) → h_0^(dec,1) */}
      <ContextProjection />

      {/* Capas del decoder desdobladas en el tiempo */}
      <div className="flex flex-col gap-2 overflow-x-auto py-5 px-3">
        <DecoderLayer
          layerNum={1}
          decTimesteps={decTimesteps}
          visibleCount={visibleCount}
          activeT={activeT}
        />

        {/* Flechas verticales entre capas (datos fluyen hacia abajo) */}
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
                  <div className="text-orange-500/70 text-xs font-mono select-none leading-none">↓</div>
                </div>
              ))}
          </div>
        </div>

        <DecoderLayer
          layerNum={2}
          decTimesteps={decTimesteps}
          visibleCount={visibleCount}
          activeT={activeT}
        />

        {timestep === 7 && visibleCount === 0 && (
          <WaitingMessage className="ml-16 mt-1">
            Presiona → para comenzar el decoder
          </WaitingMessage>
        )}
      </div>

      {/* Mecanismo de atención (solo en modo atención) */}
      {atencion && visibleCount > 0 && (
        <div>
          <SectionHeader title="Mecanismo de atención" color="#fbbf24" subtitle="Luong general" size="sm" />
          <AttentionLayer />
        </div>
      )}

      {/* Proyección sobre vocabulario */}
      {visibleCount > 0 && (
        <div>
          <SectionHeader title="Proyección sobre vocabulario" color="#fb923c" subtitle="softmax sobre 12 tokens" size="sm" />
          <SoftmaxHead />
        </div>
      )}
    </div>
  );
}
