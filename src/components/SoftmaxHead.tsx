import { motion, AnimatePresence } from 'framer-motion';
import { useStore } from '../state/store';
import { getDecoderTimesteps, getTranslationScenario } from '../data/index';
import { appData } from '../data/index';
import { hasAttention } from '../data/types';
import { decoderTokenColor } from '../utils/colors';
import { fmtPct, fmt } from '../utils/math-format';
import { durationSec } from '../utils/timing';
import MatrixDisplay from './MatrixDisplay';

export default function SoftmaxHead() {
  const { timestep, arquitectura, atencion, velocidad } = useStore();
  const decStep = timestep - 7; // 1..6
  const visible = decStep >= 1 && decStep <= 6;

  const decTimesteps = getDecoderTimesteps(arquitectura, atencion);
  const scenario = getTranslationScenario(arquitectura, atencion);
  const vocab = appData.config.vocab;

  if (!visible) return null;

  const tsData = decTimesteps[decStep - 1];
  const { logits, probas, argmax } = tsData.softmax;
  const maxP = Math.max(...probas);
  const W_out = scenario.decoder.W_out;

  // Vector que realmente entra a W_out: h_dec_t^(2) sin atención, h̃_t con atención.
  const inputVec = hasAttention(tsData) ? tsData.attention.h_tilde : tsData.layer2.h_t;
  const inputLabel = hasAttention(tsData) ? 'h̃_t' : 'h_dec_t^(2)';
  const inputSubtitle = hasAttention(tsData)
    ? 'sale del mecanismo de atención'
    : 'estado oculto capa 2 decoder';
  const inputColor = hasAttention(tsData) ? '#f97316' : '#fb923c';

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={decStep}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0 }}
        transition={{ duration: durationSec('embeddingAparece', velocidad) }}
        className="flex flex-row gap-5 items-start overflow-x-auto pb-2"
      >
        {/* Vector de entrada al softmax (h_dec o h̃_t) */}
        <div
          className="flex flex-col gap-1 shrink-0 p-3 rounded-lg bg-gray-900 border"
          style={{ borderColor: hasAttention(tsData) ? 'rgba(249,115,22,0.5)' : 'rgb(31,41,55)' }}
        >
          <div className="flex items-center gap-1.5">
            <span className="font-mono text-[11px] font-semibold" style={{ color: inputColor }}>
              {inputLabel}
            </span>
            {hasAttention(tsData) && (
              <span className="text-[8px] font-mono uppercase tracking-wider text-amber-400/80 bg-amber-950/40 px-1.5 py-0.5 rounded border border-amber-900/40">
                atención
              </span>
            )}
          </div>
          <span className="font-mono text-[9px] text-gray-500">{inputSubtitle} (dim {inputVec.length})</span>
          <div className="flex flex-col gap-0.5 mt-1">
            {inputVec.map((v, i) => (
              <span key={i} className="font-mono text-[10px]" style={{ color: inputColor, opacity: 0.85 }}>
                {fmt(v)}
              </span>
            ))}
          </div>
        </div>

        <div className="self-center text-gray-600 font-mono text-sm select-none mt-4">×</div>

        {/* W_out */}
        <div className="flex flex-col gap-1 shrink-0 p-3 rounded-lg bg-gray-900 border border-gray-800">
          <span className="font-mono text-[11px] font-semibold text-orange-400">W_out</span>
          <span className="font-mono text-[9px] text-gray-500">(12 × 3) → logits</span>
          <MatrixDisplay matrix={W_out} label="" />
        </div>

        <div className="self-center text-gray-600 font-mono text-sm select-none mt-4">→</div>

        {/* Softmax distribution */}
        <div className="flex flex-col gap-1 p-3 rounded-lg bg-gray-900 border border-gray-800 shrink-0">
          <span className="font-mono text-[11px] font-semibold text-orange-400">Softmax (vocabulario)</span>
          <span className="font-mono text-[9px] text-gray-500">paso decoder t={decStep}</span>
          <div className="flex flex-col gap-1.5 mt-1.5">
            {vocab.map((tok, i) => {
              const p = probas[i];
              const lg = logits[i];
              const isWinner = tok === argmax;
              const color = decoderTokenColor(tok);
              return (
                <div key={tok} className="flex items-center gap-2">
                  <span
                    className="font-mono text-[10px] w-16 text-right shrink-0"
                    style={{ color: isWinner ? color : '#6b7280' }}
                  >
                    {tok}
                  </span>
                  <span className="font-mono text-[9px] text-gray-600 w-12 text-right shrink-0">
                    {fmt(lg)}
                  </span>
                  <div className="w-28 h-3 bg-gray-800 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${(p / maxP) * 100}%` }}
                      transition={{ duration: durationSec('vectorViajando', velocidad), ease: 'easeOut' }}
                      style={{ backgroundColor: isWinner ? color : '#374151' }}
                    />
                  </div>
                  <span
                    className="font-mono text-[10px] w-10"
                    style={{ color: isWinner ? color : '#4b5563' }}
                  >
                    {fmtPct(p)}
                  </span>
                  {isWinner && (
                    <motion.span
                      initial={{ opacity: 0, scale: 0.5 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="text-[9px] font-bold"
                      style={{ color }}
                    >
                      ← elegido
                    </motion.span>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Token generado + hint autoregresivo */}
        <div className="self-center flex flex-col gap-2 p-3 rounded-lg bg-gray-900/60 border border-dashed border-orange-900/50 shrink-0 max-w-[180px]">
          <span className="font-mono text-[10px] font-semibold text-orange-400">Token generado</span>
          <motion.div
            key={argmax}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="font-mono text-lg font-bold"
            style={{ color: decoderTokenColor(argmax) }}
          >
            "{argmax}"
          </motion.div>
          {decStep < 6 && (
            <span className="text-[9px] text-gray-600 font-mono leading-snug">
              → se convierte en el embedding de entrada del paso t={decStep + 1}
            </span>
          )}
          {decStep === 6 && (
            <span className="text-[9px] text-orange-700 font-mono leading-snug">
              &lt;END&gt; — el decoder se detiene
            </span>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
