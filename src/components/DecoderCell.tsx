import { useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { X, Microscope } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useStore } from '../state/store';
import { decoderTokenColor } from '../utils/colors';
import { fmt } from '../utils/math-format';
import { durationSec } from '../utils/timing';
import type { EncoderLayerState, EncoderLayerStateLstm } from '../data/types';

const TOOLTIP_WIDTH = 230;
const VIEWPORT_MARGIN = 8;

interface Props {
  layer: 1 | 2;
  t: number;
  layerState: EncoderLayerState;
  inputToken: string;
  outputToken: string;
  isActive: boolean;
  tooltipSide: 'above' | 'below';
  showCellState?: boolean;
}

const sup = (l: 1 | 2) => (l === 1 ? '⁽¹⁾' : '⁽²⁾');

export default function DecoderCell({
  layer, t, layerState, inputToken, outputToken, isActive, tooltipSide, showCellState = false
}: Props) {
  const [hovered, setHovered] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const cellRef = useRef<HTMLDivElement>(null);
  const { fijarTooltip, liberarTooltip, tooltipsFijos, abrirModalCelda, velocidad } = useStore();
  const id = `dec-l${layer}-t${t}`;
  const fixed = tooltipsFijos.includes(id);
  const showTooltip = hovered || fixed;
  const color = decoderTokenColor(outputToken);

  const { h_prev, x_t, h_t } = layerState;
  const lstmState = showCellState ? (layerState as EncoderLayerStateLstm) : null;
  const label = `h${t}${sup(layer)}`;
  const prevLabel = t === 1 ? `h₀${sup(layer)}` : `h${t - 1}${sup(layer)}`;

  useLayoutEffect(() => {
    if (!showTooltip || !cellRef.current) return;
    const recalc = () => {
      const r = cellRef.current!.getBoundingClientRect();
      const half = TOOLTIP_WIDTH / 2;
      const ideal = r.left + r.width / 2;
      const minL = VIEWPORT_MARGIN + half;
      const maxL = window.innerWidth - VIEWPORT_MARGIN - half;
      setPos({
        top: tooltipSide === 'above' ? r.top - 8 : r.bottom + 8,
        left: Math.max(minL, Math.min(maxL, ideal)),
      });
    };
    recalc();
    window.addEventListener('scroll', recalc, true);
    window.addEventListener('resize', recalc);
    return () => {
      window.removeEventListener('scroll', recalc, true);
      window.removeEventListener('resize', recalc);
    };
  }, [showTooltip, tooltipSide]);

  return (
    <div className="flex flex-col items-center">
      <motion.div
        ref={cellRef}
        initial={{ opacity: 0, scale: 0.7 }}
        animate={{ opacity: 1, scale: isActive ? 1.05 : 1 }}
        transition={{ duration: durationSec('desdoblamientoTimestep', velocidad), ease: 'easeOut' }}
        className="w-24 rounded-lg border-2 p-2 cursor-pointer select-none flex flex-col gap-1.5"
        style={{
          borderColor: isActive ? color : `${color}66`,
          backgroundColor: `${color}${isActive ? '22' : '0e'}`,
          boxShadow: isActive ? `0 0 14px ${color}55` : undefined,
        }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        onClick={() => (fixed ? liberarTooltip(id) : fijarTooltip(id))}
      >
        <div className="font-mono text-[10px] font-semibold" style={{ color: `${color}cc` }}>
          {label}
        </div>
        <div className="font-mono text-[8px] text-gray-500 truncate">← "{inputToken}"</div>
        <div className="flex flex-wrap gap-x-1.5 gap-y-0.5">
          {h_t.map((v, i) => (
            <span key={i} className="font-mono text-[9px]" style={{ color: `${color}aa` }}>
              {v.toFixed(2)}
            </span>
          ))}
        </div>
        {lstmState && (
          <>
            <div className="font-mono text-[8px] font-semibold text-yellow-500/70 mt-0.5">
              C{t}{sup(layer)}
            </div>
            <div className="flex flex-wrap gap-x-1.5 gap-y-0.5">
              {lstmState.c_t.map((v, i) => (
                <span key={i} className="font-mono text-[9px] text-yellow-400/60">
                  {v.toFixed(2)}
                </span>
              ))}
            </div>
          </>
        )}
      </motion.div>

      {createPortal(
        <AnimatePresence>
          {showTooltip && pos && (
            <motion.div
              key="tt"
              initial={{ opacity: 0, y: tooltipSide === 'above' ? 4 : -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: tooltipSide === 'above' ? 4 : -4 }}
              transition={{ duration: durationSec('cambioTooltip', velocidad) }}
              className="fixed z-[100] bg-gray-900 border rounded-xl p-3 shadow-2xl pointer-events-auto"
              style={{
                top: pos.top,
                left: pos.left,
                width: TOOLTIP_WIDTH,
                transform: tooltipSide === 'above'
                  ? 'translate(-50%, -100%)'
                  : 'translate(-50%, 0)',
                borderColor: `${color}44`,
              }}
            >
            <div className="flex items-center justify-between mb-2.5">
              <span className="font-mono text-[11px] font-semibold" style={{ color }}>
                {label} (dec) — "{outputToken}"
              </span>
              {fixed && (
                <button
                  className="text-gray-500 hover:text-gray-200 ml-2"
                  onClick={(e) => { e.stopPropagation(); liberarTooltip(id); }}
                >
                  <X size={11} />
                </button>
              )}
            </div>

            <div className="flex flex-col gap-2">
              <TooltipRow label={`emb("${inputToken}")`} values={x_t} color="#94a3b8" />
              <TooltipRow label={prevLabel} values={h_prev} color={`${color}88`} />
              {lstmState && (
                <TooltipRow
                  label={`C_${t - 1}${sup(layer)}`}
                  values={lstmState.c_prev}
                  color="#ca8a04"
                />
              )}
              <div className="border-t border-gray-700 pt-2">
                <TooltipRow label={label} values={h_t} color={color} highlight />
              </div>
              {lstmState && (
                <TooltipRow
                  label={`C_${t}${sup(layer)}`}
                  values={lstmState.c_t}
                  color="#facc15"
                  highlight
                />
              )}
            </div>

            <div className="flex items-center justify-between mt-2 pt-2 border-t border-gray-700/50">
              {!fixed && <span className="text-gray-700 text-[9px]">click para fijar</span>}
              <button
                className="flex items-center gap-1 text-[9px] text-orange-500 hover:text-orange-300 transition-colors ml-auto"
                onClick={(e) => {
                  e.stopPropagation();
                  abrirModalCelda({ layer, t, lado: 'dec' });
                }}
              >
                <Microscope size={10} /> Ver interior
              </button>
            </div>
            </motion.div>
          )}
        </AnimatePresence>,
        document.body
      )}
    </div>
  );
}

function TooltipRow({
  label, values, color, highlight,
}: {
  label: string; values: number[]; color: string; highlight?: boolean;
}) {
  return (
    <div className={`flex flex-col gap-0.5 ${highlight ? 'bg-gray-800/70 rounded-md p-1.5' : ''}`}>
      <span className="text-gray-500 font-mono text-[10px]">{label}</span>
      <div className="flex flex-wrap gap-x-2">
        {values.map((v, i) => (
          <span key={i} className="font-mono text-[11px]" style={{ color }}>
            {fmt(v)}
          </span>
        ))}
      </div>
    </div>
  );
}
