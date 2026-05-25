import { motion, AnimatePresence } from 'framer-motion';
import { useStore } from '../state/store';
import { getTranslationScenario } from '../data/index';
import MatrixDisplay from './MatrixDisplay';
import { fmt } from '../utils/math-format';
import { durationSec } from '../utils/timing';

export default function ContextProjection() {
  const { timestep, arquitectura, atencion, velocidad } = useStore();
  const visible = timestep >= 7;

  const scenario = getTranslationScenario(arquitectura, atencion);
  const ctx = scenario.context;

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: durationSec('embeddingAparece', velocidad) }}
          className="flex flex-row items-start gap-4 p-3 rounded-lg bg-gray-900/80 border border-gray-800 overflow-x-auto shrink-0"
        >
          {/* h_T^(2) */}
          <ColSection title="h₇⁽²⁾" subtitle="encoder final (dim l=4)" color="#818cf8">
            <div className="flex flex-col gap-0.5 mt-1">
              {ctx.h_T_layer2.map((v, i) => (
                <span key={i} className="font-mono text-[10px] text-indigo-300">{fmt(v)}</span>
              ))}
            </div>
          </ColSection>

          <Sym label="×" />

          {/* W_c */}
          <ColSection title="W_c" subtitle="proyección (5 × 4)" color="#fb923c">
            <MatrixDisplay matrix={ctx.W_c} label="" />
          </ColSection>

          <Sym label="=" />

          {/* h_0_decoder_layer1 */}
          <ColSection title="h₀^(dec,1)" subtitle="estado inicial capa 1 dec. (dim p=5)" color="#4ade80">
            <div className="flex flex-col gap-0.5 mt-1">
              {ctx.h_0_decoder_layer1.map((v, i) => (
                <span key={i} className="font-mono text-[10px] text-green-400">{fmt(v)}</span>
              ))}
            </div>
          </ColSection>

          <div className="self-center border-l border-gray-700 pl-4 ml-2 flex flex-col gap-1">
            <span className="text-[9px] text-gray-500 font-mono">capa 2 del decoder</span>
            <span className="text-[9px] text-gray-600 font-mono">h₀^(dec,2) = [0…0]</span>
          </div>

          {atencion && (
            <div className="self-center ml-2 pl-3 border-l border-amber-900/40 max-w-[260px]">
              <span className="text-[9px] font-mono text-amber-400/80 uppercase tracking-wider">
                Nota · con atención
              </span>
              <p className="text-[10px] text-gray-400 leading-snug mt-0.5">
                El estado inicial del decoder es el <span className="text-green-400 font-mono">mismo</span> con o sin
                atención. La atención no reemplaza esta inicialización — agrega información{' '}
                <span className="text-amber-300">en cada paso de salida</span> (ver más abajo).
              </p>
            </div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function ColSection({ title, subtitle, color, children }: {
  title: string; subtitle: string; color: string; children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-0.5 shrink-0">
      <span className="font-mono text-[11px] font-semibold" style={{ color }}>{title}</span>
      <span className="font-mono text-[9px] text-gray-500">{subtitle}</span>
      {children}
    </div>
  );
}

function Sym({ label }: { label: string }) {
  return (
    <div className="self-center text-gray-600 font-mono text-sm px-1 select-none mt-5">{label}</div>
  );
}
