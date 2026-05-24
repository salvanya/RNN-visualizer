import { useState } from 'react';
import { X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useStore } from '../state/store';
import { encoderTokenColor } from '../utils/colors';
import { fmt } from '../utils/math-format';
import type { EncoderLayerState } from '../data/types';

interface Props {
  layer: 1 | 2;
  t: number;
  layerState: EncoderLayerState;
  token: string;
  isActive: boolean;
  /** Where the tooltip appears relative to the cell */
  tooltipSide: 'above' | 'below';
}

// Superscript notation for layer number
const sup = (l: 1 | 2) => (l === 1 ? '⁽¹⁾' : '⁽²⁾');

export default function Cell({ layer, t, layerState, token, isActive, tooltipSide }: Props) {
  const [hovered, setHovered] = useState(false);
  const { fijarTooltip, liberarTooltip, tooltipsFijos } = useStore();
  const id = `enc-l${layer}-t${t}`;
  const fixed = tooltipsFijos.includes(id);
  const showTooltip = hovered || fixed;
  const color = encoderTokenColor(token);

  const { h_prev, x_t, h_t } = layerState;
  const label = `h${t}${sup(layer)}`;
  const prevLabel = `h${t - 1}${sup(layer)}`;

  const tooltipPositionClass =
    tooltipSide === 'above' ? 'bottom-full mb-2' : 'top-full mt-2';

  return (
    <div className="relative flex flex-col items-center">
      <motion.div
        initial={{ opacity: 0, scale: 0.7 }}
        animate={{
          opacity: 1,
          scale: isActive ? 1.05 : 1,
        }}
        transition={{ duration: 0.35, ease: 'easeOut' }}
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
        {/* Label */}
        <div className="font-mono text-[10px] font-semibold" style={{ color: `${color}cc` }}>
          {label}
        </div>
        {/* h_t values */}
        <div className="flex flex-wrap gap-x-1.5 gap-y-0.5">
          {h_t.map((v, i) => (
            <span key={i} className="font-mono text-[9px]" style={{ color: `${color}aa` }}>
              {v.toFixed(2)}
            </span>
          ))}
        </div>
      </motion.div>

      {/* Tooltip */}
      <AnimatePresence>
        {showTooltip && (
          <motion.div
            initial={{ opacity: 0, y: tooltipSide === 'above' ? 4 : -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: tooltipSide === 'above' ? 4 : -4 }}
            transition={{ duration: 0.15 }}
            className={`absolute ${tooltipPositionClass} z-50 bg-gray-900 border rounded-xl p-3 shadow-2xl min-w-[210px]`}
            style={{ borderColor: `${color}44` }}
          >
            <div className="flex items-center justify-between mb-2.5">
              <span className="font-mono text-[11px] font-semibold" style={{ color }}>
                {label} — "{token}"
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
              <TooltipRow label={`x_${t} (embedding)`} values={x_t} color="#94a3b8" />
              <TooltipRow label={prevLabel} values={h_prev} color={`${color}88`} />
              <div className="border-t border-gray-700 pt-2">
                <TooltipRow label={label} values={h_t} color={color} highlight />
              </div>
            </div>

            {!fixed && (
              <div className="text-gray-700 text-[9px] mt-2 text-center">click para fijar</div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function TooltipRow({
  label,
  values,
  color,
  highlight,
}: {
  label: string;
  values: number[];
  color: string;
  highlight?: boolean;
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
