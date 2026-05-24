import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useStore } from '../state/store';
import { getTranslationScenario, appData } from '../data/index';
import { hasAttention } from '../data/types';
import MatrixDisplay from './MatrixDisplay';
import VectorDisplay from './VectorDisplay';
import { encoderTokenColor } from '../utils/colors';
import { fmt } from '../utils/math-format';

export default function AttentionLayer() {
  const { timestep, arquitectura, atencion } = useStore();
  const [showWa, setShowWa] = useState(false);

  if (!atencion) return null;

  const decIdx = timestep - 8; // 0-indexed decoder step
  if (decIdx < 0 || decIdx > 5) return null;

  const scenario = getTranslationScenario(arquitectura, true);
  const ts = scenario.decoder.timesteps[decIdx];

  if (!hasAttention(ts)) return null;

  const { attention, softmax: sm } = ts;
  const { scores, alphas, contexto, h_tilde } = attention;
  const encoderTokens = appData.config.encoderTokens;
  const W_a = scenario.decoder.W_a!;

  const peakIdx = alphas.indexOf(Math.max(...alphas));
  const maxScore = Math.max(...scores.map(Math.abs));

  const inputToken = ts.input_token;
  const genToken = sm.argmax;

  return (
    <div className="border border-amber-800/40 rounded-lg bg-amber-950/10 p-3 space-y-3">
      {/* Encabezado */}
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-amber-400/80">
          Atención Luong — paso {decIdx + 1}
          <span className="text-gray-500 font-normal normal-case ml-2">
            (entrada: <span className="text-amber-300">{inputToken}</span>
            {' '}→ gen: <span className="text-amber-200 font-semibold">{genToken}</span>)
          </span>
        </h3>
        <button
          onClick={() => setShowWa(v => !v)}
          className="text-[10px] text-gray-500 hover:text-gray-300 font-mono border border-gray-700 rounded px-2 py-0.5 transition-colors shrink-0"
        >
          {showWa ? 'ocultar W_a' : 'ver W_a (3×4)'}
        </button>
      </div>

      {/* Scores y Alphas — barras horizontales */}
      <div className="flex gap-4 flex-wrap">
        {/* Columna: scores e_{t,i} */}
        <div className="flex flex-col gap-1 min-w-0">
          <div className="text-[10px] text-gray-500 font-mono mb-0.5">
            Scores &nbsp;<span className="text-gray-600">e_&#123;t,i&#125; = h_dec^T W_a h_enc_i</span>
          </div>
          <div className="flex flex-col gap-0.5">
            {scores.map((s, i) => {
              const color = encoderTokenColor(encoderTokens[i]);
              const barW = maxScore > 0 ? Math.abs(s) / maxScore * 80 : 0;
              return (
                <div key={i} className="flex items-center gap-1.5">
                  <span
                    className="font-mono text-[9px] w-12 text-right shrink-0"
                    style={{ color }}
                  >
                    {encoderTokens[i]}
                  </span>
                  <div className="flex items-center gap-0.5">
                    {s < 0 && (
                      <div
                        className="h-2 rounded-sm"
                        style={{ width: `${barW}px`, backgroundColor: '#f87171', opacity: 0.7 }}
                      />
                    )}
                    {s >= 0 && (
                      <div
                        className="h-2 rounded-sm"
                        style={{ width: `${barW}px`, backgroundColor: color, opacity: 0.7 }}
                      />
                    )}
                  </div>
                  <span className="font-mono text-[9px] text-gray-400">{fmt(s)}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Separador */}
        <div className="text-gray-700 text-sm self-center">→ softmax →</div>

        {/* Columna: alphas α_{t,i} — barras proporcionales */}
        <div className="flex flex-col gap-1 min-w-0">
          <div className="text-[10px] text-gray-500 font-mono mb-0.5">
            Pesos α_&#123;t,i&#125; &nbsp;<span className="text-gray-600">(softmax de scores)</span>
          </div>
          <div className="flex items-end gap-1">
            {alphas.map((alpha, i) => {
              const color = encoderTokenColor(encoderTokens[i]);
              const isPeak = i === peakIdx;
              return (
                <div key={i} className="flex flex-col items-center gap-0.5" style={{ width: '46px' }}>
                  <motion.div
                    className="w-full rounded-t-sm relative"
                    style={{
                      height: `${Math.max(3, alpha * 72)}px`,
                      backgroundColor: color,
                      opacity: 0.25 + alpha * 0.75,
                    }}
                    initial={{ height: 0 }}
                    animate={{ height: `${Math.max(3, alpha * 72)}px` }}
                    transition={{ duration: 0.4, ease: 'easeOut' }}
                  >
                    {isPeak && (
                      <motion.div
                        className="absolute inset-0 rounded-t-sm"
                        style={{ boxShadow: `0 0 4px ${color}` }}
                        animate={{ opacity: [0.4, 1, 0.4] }}
                        transition={{ duration: 1.8, repeat: Infinity }}
                      />
                    )}
                  </motion.div>
                  <span className={`font-mono text-[9px] ${isPeak ? 'font-bold' : ''}`} style={{ color: isPeak ? color : '#6b7280' }}>
                    {alpha.toFixed(3)}
                  </span>
                  <span
                    className="font-mono text-[8px] truncate w-full text-center"
                    style={{ color }}
                  >
                    {encoderTokens[i]}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="text-[9px] text-amber-400/50 font-mono mt-1">
            ↑ pico: "<span className="text-amber-300">{encoderTokens[peakIdx]}</span>"
            &nbsp;α={alphas[peakIdx].toFixed(3)}
          </div>
        </div>

        {/* Separador */}
        <div className="text-gray-700 text-sm self-center">→</div>

        {/* Columna: contexto c_t y h_tilde */}
        <div className="flex gap-4 flex-wrap">
          <VectorDisplay
            values={contexto}
            label={`c_t = Σ αᵢ h_enc_i  (dim ${contexto.length})`}
            color="#fbbf24"
            withBars
          />
          <VectorDisplay
            values={h_tilde}
            label={`h̃_t = tanh(W_combine·[h_dec;c_t])  (dim ${h_tilde.length})`}
            color="#f97316"
            withBars
          />
        </div>
      </div>

      {/* W_a expandible */}
      <AnimatePresence>
        {showWa && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="border-t border-amber-900/30 pt-2 mt-1">
              <MatrixDisplay
                matrix={W_a}
                label="W_a (Q×L = 3×4) — atención Luong general: e_{t,i} = h_dec^T W_a h_enc_i"
                rowLabels={['q₀', 'q₁', 'q₂']}
                colLabels={['l₀', 'l₁', 'l₂', 'l₃']}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Fórmula resumen */}
      <div className="text-[9px] text-gray-600 font-mono border-t border-gray-800 pt-2">
        softmax sobre vocab: <span className="text-gray-500">W_out · h̃_t</span>
        {' '}→ argmax = <span className="text-amber-300 font-semibold">{genToken}</span>
        {' '}(p={sm.probas[sm.probas.indexOf(Math.max(...sm.probas))].toFixed(3)})
      </div>
    </div>
  );
}
