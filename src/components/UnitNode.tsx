import { useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useStore } from '../state/store';
import { fmt } from '../utils/math-format';
import { durationSec } from '../utils/timing';

const TOOLTIP_WIDTH = 200;
const VIEWPORT_MARGIN = 8;

export interface UnitTooltipRow {
  label: string;
  value: number;
  color: string;
  highlight?: boolean;
}

interface Props {
  id: string;                       // unique id for fixable tooltip
  cx: number;                        // center x (absolute within canvas)
  cy: number;                        // center y
  size: number;
  value: number;                    // primary scalar to display
  borderColor: string;              // border tint
  fillOpacity?: number;             // background tint strength (0..1)
  cValue?: number;                  // LSTM C scalar (optional)
  isActive?: boolean;
  label: string;                    // e.g. "h₃⁽¹⁾[2]"
  tokenLabel?: string;               // e.g. "encantó" — shown in tooltip header
  tooltipRows: UnitTooltipRow[];     // breakdown shown in tooltip
  tooltipSide?: 'above' | 'below';
}

export default function UnitNode({
  id,
  cx,
  cy,
  size,
  value,
  borderColor,
  fillOpacity = 0.13,
  cValue,
  isActive,
  label,
  tokenLabel,
  tooltipRows,
  tooltipSide = 'above',
}: Props) {
  const [hovered, setHovered] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const nodeRef = useRef<HTMLDivElement>(null);
  const { fijarTooltip, liberarTooltip, tooltipsFijos, velocidad } = useStore();
  const fixed = tooltipsFijos.includes(id);
  const showTooltip = hovered || fixed;

  const hasC = cValue !== undefined;

  useLayoutEffect(() => {
    if (!showTooltip || !nodeRef.current) return;
    const recalc = () => {
      const r = nodeRef.current!.getBoundingClientRect();
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
    <>
      <motion.div
        ref={nodeRef}
        initial={{ opacity: 0, scale: 0.6 }}
        animate={{ opacity: 1, scale: isActive ? 1.08 : 1 }}
        transition={{ duration: durationSec('desdoblamientoTimestep', velocidad), ease: 'easeOut' }}
        className="absolute rounded-md cursor-pointer select-none flex flex-col items-center justify-center"
        style={{
          left: cx - size / 2,
          top: cy - size / 2,
          width: size,
          height: size,
          padding: 2,
          border: `1.5px solid ${borderColor}${isActive ? 'ff' : 'aa'}`,
          background: `#0a0a0acc`,
          boxShadow: isActive
            ? `0 0 10px ${borderColor}66, inset 0 0 0 1px ${borderColor}${Math.round(fillOpacity * 255).toString(16).padStart(2, '0')}`
            : `inset 0 0 0 1px ${borderColor}${Math.round(fillOpacity * 255).toString(16).padStart(2, '0')}`,
          zIndex: 5,
        }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        onClick={() => (fixed ? liberarTooltip(id) : fijarTooltip(id))}
      >
        <span
          className="font-mono leading-none"
          style={{
            color: borderColor,
            fontSize: hasC ? '8.5px' : size < 28 ? '8.5px' : '10px',
          }}
        >
          {value.toFixed(2)}
        </span>
        {hasC && (
          <>
            <div
              className="my-0.5"
              style={{
                width: size - 8,
                height: 1,
                background: `${borderColor}55`,
              }}
            />
            <span
              className="font-mono leading-none text-yellow-400/85"
              style={{ fontSize: '8px' }}
            >
              {cValue!.toFixed(2)}
            </span>
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
              className={`fixed z-[100] bg-gray-900 border rounded-xl p-2.5 shadow-2xl ${fixed ? 'pointer-events-auto' : 'pointer-events-none'}`}
              style={{
                top: pos.top,
                left: pos.left,
                width: TOOLTIP_WIDTH,
                transform: tooltipSide === 'above'
                  ? 'translate(-50%, -100%)'
                  : 'translate(-50%, 0)',
                borderColor: `${borderColor}44`,
              }}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-mono text-[10px] font-semibold" style={{ color: borderColor }}>
                  {label}{tokenLabel ? ` · "${tokenLabel}"` : ''}
                </span>
                {fixed && (
                  <button
                    className="text-gray-500 hover:text-gray-200 ml-2"
                    onClick={(e) => { e.stopPropagation(); liberarTooltip(id); }}
                  >
                    <X size={10} />
                  </button>
                )}
              </div>

              <div className="flex flex-col gap-1">
                {tooltipRows.map((row, i) => (
                  <div
                    key={i}
                    className={`flex items-center justify-between gap-2 px-1.5 py-0.5 rounded ${row.highlight ? 'bg-gray-800/70' : ''}`}
                  >
                    <span className="text-gray-500 font-mono text-[9px]">{row.label}</span>
                    <span className="font-mono text-[10px]" style={{ color: row.color }}>
                      {fmt(row.value)}
                    </span>
                  </div>
                ))}
              </div>

              {!fixed && (
                <div className="text-gray-700 text-[8px] mt-1.5 pt-1.5 border-t border-gray-700/40 text-center">
                  click para fijar
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>,
        document.body
      )}
    </>
  );
}
