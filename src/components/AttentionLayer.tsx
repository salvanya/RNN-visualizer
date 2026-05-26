import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useStore } from '../state/store';
import { getTranslationScenario, appData } from '../data/index';
import { hasAttention, isBahdanauAttention, type AttentionOutput } from '../data/types';
import MatrixDisplay from './MatrixDisplay';
import VectorDisplay from './VectorDisplay';
import { encoderTokenColor } from '../utils/colors';
import { fmt } from '../utils/math-format';
import { durationSec } from '../utils/timing';

export default function AttentionLayer() {
  const { timestep, arquitectura, atencion, velocidad } = useStore();
  const [showWa, setShowWa] = useState(false);

  if (atencion === "none") return null;

  const decIdx = timestep - 8;
  if (decIdx < 0 || decIdx > 5) return null;

  const scenario = getTranslationScenario(arquitectura, atencion);
  const ts = scenario.decoder.timesteps[decIdx];

  if (!hasAttention(ts)) return null;

  const { attention, softmax: sm } = ts;
  const encoderTokens = appData.config.encoderTokens;
  const isBahdanau = isBahdanauAttention(attention);
  const variantLabel = isBahdanau ? "Bahdanau (aditiva)" : "Luong (multiplicativa)";

  const inputToken = ts.input_token;
  const genToken = sm.argmax;

  return (
    <div className="border border-amber-800/40 rounded-lg bg-amber-950/10 p-3 space-y-3">
      {/* Encabezado */}
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-amber-400/80">
          Atención {variantLabel} — paso {decIdx + 1}
          <span className="text-gray-500 font-normal normal-case ml-2">
            (entrada: <span className="text-amber-300">{inputToken}</span>
            {' '}→ gen: <span className="text-amber-200 font-semibold">{genToken}</span>)
          </span>
        </h3>
        <button
          onClick={() => setShowWa(v => !v)}
          className="text-[10px] text-gray-500 hover:text-gray-300 font-mono border border-gray-700 rounded px-2 py-0.5 transition-colors shrink-0"
        >
          {showWa ? 'ocultar pesos' : isBahdanau ? 'ver W_a, U_a, v_a' : 'ver W_a (3×4)'}
        </button>
      </div>

      {isBahdanau ? (
        <BahdanauBody attention={attention} velocidad={velocidad} />
      ) : (
        <LuongBody attention={attention} velocidad={velocidad} />
      )}

      {/* Matrices expandibles */}
      <AnimatePresence>
        {showWa && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: durationSec('aperturaModal', velocidad) }}
            className="overflow-hidden"
          >
            <div className="border-t border-amber-900/30 pt-2 mt-1 flex flex-wrap gap-4">
              {isBahdanau ? (
                <BahdanauWeights scenario={scenario} />
              ) : (
                <MatrixDisplay
                  matrix={scenario.decoder.W_a!}
                  label="W_a (Q×L = 3×4) — Luong: e_{t,i} = h_dec^T W_a h_enc_i"
                  rowLabels={['q₀', 'q₁', 'q₂']}
                  colLabels={['l₀', 'l₁', 'l₂', 'l₃']}
                />
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Fórmula resumen */}
      <div className="text-[9px] text-gray-600 font-mono border-t border-gray-800 pt-2">
        {isBahdanau ? (
          <>
            entrada al decoder L1: <span className="text-amber-300">[emb({inputToken}); c_t]</span>
            {' '}(dim {appData.config.d}+{appData.config.l}={appData.config.d + appData.config.l});
            {' '}salida: <span className="text-gray-500">softmax(W_out · h_t^(2))</span>
            {' '}→ <span className="text-amber-300 font-semibold">{genToken}</span>
          </>
        ) : (
          <>
            softmax sobre vocab: <span className="text-gray-500">W_out · h̃_t</span>
            {' '}→ <span className="text-amber-300 font-semibold">{genToken}</span>
            {' '}(p={Math.max(...sm.probas).toFixed(3)})
          </>
        )}
      </div>
    </div>
  );
}

// ─── Cuerpo común: scores + alphas + contexto ─────────────────────────────

function ScoresAlphas({
  scores,
  alphas,
  velocidad,
  scoreFormula,
}: {
  scores: number[];
  alphas: number[];
  velocidad: 0.5 | 1 | 2;
  scoreFormula: React.ReactNode;
}) {
  const encoderTokens = appData.config.encoderTokens;
  const peakIdx = alphas.indexOf(Math.max(...alphas));
  const maxScore = Math.max(...scores.map(Math.abs)) || 1;

  return (
    <div className="flex gap-4 flex-wrap">
      {/* Scores */}
      <div className="flex flex-col gap-1 min-w-0">
        <div className="text-[10px] text-gray-500 font-mono mb-0.5">
          Scores &nbsp;<span className="text-gray-600">{scoreFormula}</span>
        </div>
        <div className="flex flex-col gap-0.5">
          {scores.map((s, i) => {
            const color = encoderTokenColor(encoderTokens[i]);
            const barW = (Math.abs(s) / maxScore) * 80;
            return (
              <div key={i} className="flex items-center gap-1.5">
                <span className="font-mono text-[9px] w-12 text-right shrink-0" style={{ color }}>
                  {encoderTokens[i]}
                </span>
                <div className="flex items-center gap-0.5">
                  <div
                    className="h-2 rounded-sm"
                    style={{
                      width: `${barW}px`,
                      backgroundColor: s < 0 ? '#f87171' : color,
                      opacity: 0.7,
                    }}
                  />
                </div>
                <span className="font-mono text-[9px] text-gray-400">{fmt(s)}</span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="text-gray-700 text-sm self-center">→ softmax →</div>

      {/* Alphas */}
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
                  transition={{ duration: durationSec('vectorViajando', velocidad), ease: 'easeOut' }}
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
                <span className="font-mono text-[8px] truncate w-full text-center" style={{ color }}>
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
    </div>
  );
}

// ─── Luong body ───────────────────────────────────────────────────────────

function LuongBody({ attention, velocidad }: { attention: AttentionOutput; velocidad: 0.5 | 1 | 2 }) {
  if (isBahdanauAttention(attention)) return null;
  const { scores, alphas, contexto, h_tilde } = attention;

  return (
    <div className="space-y-3">
      <ScoresAlphas
        scores={scores}
        alphas={alphas}
        velocidad={velocidad}
        scoreFormula={<>e_&#123;t,i&#125; = h_dec^T W_a h_enc_i</>}
      />
      <div className="flex gap-4 flex-wrap items-center pt-1 border-t border-amber-900/20">
        <VectorDisplay
          values={contexto}
          label={`c_t = Σ αᵢ h_enc_i  (dim ${contexto.length})`}
          color="#fbbf24"
          withBars
        />
        <div className="text-gray-700 text-sm">→</div>
        <div className="flex flex-col gap-1">
          <VectorDisplay
            values={h_tilde}
            label={`h̃_t = tanh(W_combine·[h_dec;c_t])  (dim ${h_tilde.length})`}
            color="#f97316"
            withBars
          />
          <div className="text-[9px] font-mono text-orange-300/90 leading-tight max-w-[220px] mt-0.5 border-l-2 border-orange-500/60 pl-2">
            ↓ entra a <span className="text-orange-200 font-semibold">W_out</span> en lugar de{' '}
            <span className="line-through text-gray-500">h_dec_t^(2)</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Bahdanau body ────────────────────────────────────────────────────────

function BahdanauBody({ attention, velocidad }: { attention: AttentionOutput; velocidad: 0.5 | 1 | 2 }) {
  if (!isBahdanauAttention(attention)) return null;
  const { scores, alphas, contexto, h_dec_prev } = attention;

  return (
    <div className="space-y-3">
      <div className="flex gap-3 items-start flex-wrap text-[10px]">
        <div className="border border-gray-700 rounded px-2 py-1.5 bg-gray-900/40">
          <div className="text-gray-500 font-mono mb-0.5">Query (previo)</div>
          <VectorDisplay
            values={h_dec_prev}
            label={`h_dec_{t-1}^(2)  (dim ${h_dec_prev.length})`}
            color="#fb923c"
            withBars
          />
        </div>
        <div className="text-[9px] text-gray-400 font-mono max-w-[280px] leading-snug pt-2">
          Bahdanau usa el estado decoder <em>anterior</em> al step actual. En t=1 es cero;
          a partir de t=2 evoluciona y desplaza la atención.
        </div>
      </div>

      <ScoresAlphas
        scores={scores}
        alphas={alphas}
        velocidad={velocidad}
        scoreFormula={<>e_&#123;t,i&#125; = v_a^T · tanh(W_a · h_dec_&#123;t-1&#125; + U_a · h_enc_i)</>}
      />

      <div className="flex gap-4 flex-wrap items-center pt-1 border-t border-amber-900/20">
        <VectorDisplay
          values={contexto}
          label={`c_t = Σ αᵢ h_enc_i  (dim ${contexto.length})`}
          color="#fbbf24"
          withBars
        />
        <div className="flex flex-col text-[9px] font-mono text-orange-300/90 leading-tight max-w-[300px] border-l-2 border-orange-500/60 pl-2">
          <span>↑ se concatena con el embedding del token previo</span>
          <span className="text-gray-500">[emb(y_&#123;t-1&#125;) ; c_t] → entra a decoder L1</span>
          <span className="text-amber-400/70 mt-1">⇒ modifica la dinámica del decoder</span>
        </div>
      </div>
    </div>
  );
}

// ─── Bahdanau weight panels ───────────────────────────────────────────────

function BahdanauWeights({ scenario }: { scenario: { decoder: { W_a?: number[][]; U_a?: number[][]; v_a?: number[] } } }) {
  const { W_a, U_a, v_a } = scenario.decoder;
  if (!W_a || !U_a || !v_a) return null;
  return (
    <>
      <MatrixDisplay
        matrix={W_a}
        label="W_a (A×Q = 4×3) — proyecta query h_dec_{t-1}"
        rowLabels={['a₀', 'a₁', 'a₂', 'a₃']}
        colLabels={['q₀', 'q₁', 'q₂']}
      />
      <MatrixDisplay
        matrix={U_a}
        label="U_a (A×L = 4×4) — proyecta keys h_enc_i"
        rowLabels={['a₀', 'a₁', 'a₂', 'a₃']}
        colLabels={['l₀', 'l₁', 'l₂', 'l₃']}
      />
      <VectorDisplay
        values={v_a}
        label="v_a (dim A = 4) — colapsa a escalar"
        color="#fbbf24"
        withBars
      />
    </>
  );
}
