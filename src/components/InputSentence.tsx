import { motion } from 'framer-motion';
import { useStore } from '../state/store';
import { appData, getTranslationScenario } from '../data/index';
import { hasAttention } from '../data/types';
import TokenPill from './TokenPill';
import { encoderTokenColor } from '../utils/colors';
import { durationSec } from '../utils/timing';

export default function InputSentence() {
  const { timestep, atencion, modo, arquitectura, velocidad } = useStore();
  const tokens = appData.config.encoderTokens;
  const embeddings = appData.config.embeddings;

  const activeIdx = timestep >= 1 && timestep <= 7 ? timestep - 1 : null;

  // Obtener alphas del paso de decoder activo (solo en modo atención)
  let alphas: number[] | null = null;
  const decIdx = timestep - 8;
  if (atencion && modo === 'translation' && decIdx >= 0 && decIdx <= 5) {
    const scenario = getTranslationScenario(arquitectura, true);
    const ts = scenario.decoder.timesteps[decIdx];
    if (hasAttention(ts)) {
      alphas = ts.attention.alphas;
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="text-[10px] text-gray-400 uppercase tracking-[0.12em] font-semibold">
        Entrada — español
      </div>

      {/* Tokens */}
      <div className="flex items-start gap-3 flex-wrap">
        {tokens.map((token, i) => (
          <TokenPill
            key={token}
            token={token}
            embedding={embeddings[token] ?? []}
            isActive={activeIdx === i}
            isProcessed={activeIdx !== null ? i < activeIdx : timestep > 7}
          />
        ))}
      </div>

      {/* Heatmap de atención — alineado bajo cada token */}
      {alphas && (
        <motion.div
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: durationSec('embeddingAparece', velocidad) }}
          className="flex items-end gap-3 flex-wrap"
        >
          {tokens.map((token, i) => {
            const alpha = alphas![i];
            const color = encoderTokenColor(token);
            const isPeak = alpha === Math.max(...alphas!);
            return (
              <div
                key={token}
                className="flex flex-col items-center"
                style={{ width: '64px' }}
              >
                {/* Barra de alpha */}
                <motion.div
                  className="w-full rounded-t-sm"
                  style={{
                    height: `${Math.max(2, alpha * 40)}px`,
                    backgroundColor: color,
                    opacity: 0.2 + alpha * 0.8,
                    boxShadow: isPeak ? `0 0 6px ${color}` : 'none',
                  }}
                  initial={{ height: 0 }}
                  animate={{ height: `${Math.max(2, alpha * 40)}px` }}
                  transition={{ duration: durationSec('vectorViajando', velocidad), ease: 'easeOut' }}
                />
                {/* Valor numérico */}
                <span
                  className={`font-mono text-[9px] mt-0.5 ${isPeak ? 'font-bold' : ''}`}
                  style={{ color: isPeak ? color : '#4b5563' }}
                >
                  {alpha.toFixed(3)}
                </span>
              </div>
            );
          })}
          <span className="text-[9px] text-amber-400/60 font-mono self-end ml-1">
            ← α
          </span>
        </motion.div>
      )}
    </div>
  );
}
