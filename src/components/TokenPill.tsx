import { useState } from 'react';
import { X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useStore } from '../state/store';
import { encoderTokenColor } from '../utils/colors';
import { fmt } from '../utils/math-format';

interface Props {
  token: string;
  embedding: number[];
  isActive: boolean;
  isProcessed: boolean;
}

export default function TokenPill({ token, embedding, isActive, isProcessed }: Props) {
  const [hovered, setHovered] = useState(false);
  const { fijarTooltip, liberarTooltip, tooltipsFijos } = useStore();
  const id = `token-${token}`;
  const fixed = tooltipsFijos.includes(id);
  const showTooltip = hovered || fixed;
  const color = encoderTokenColor(token);

  return (
    <div className="relative flex flex-col items-center">
      <motion.div
        animate={{
          opacity: isProcessed && !isActive ? 0.55 : 1,
          scale: isActive ? 1.12 : 1,
        }}
        transition={{ duration: 0.3 }}
        className="px-3 py-1.5 rounded-full text-sm font-semibold cursor-pointer select-none border-2"
        style={{
          borderColor: color,
          color,
          backgroundColor: `${color}1a`,
          boxShadow: isActive ? `0 0 16px ${color}70` : undefined,
        }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        onClick={() => (fixed ? liberarTooltip(id) : fijarTooltip(id))}
      >
        {token}
      </motion.div>

      <AnimatePresence>
        {showTooltip && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
            className="absolute top-full mt-2 z-50 bg-gray-900 border rounded-lg p-3 shadow-2xl min-w-[180px]"
            style={{ borderColor: `${color}55` }}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-mono text-[11px] font-semibold" style={{ color }}>
                embed("{token}")
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

            <div className="flex flex-col gap-1">
              {embedding.map((v, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-gray-600 font-mono text-[10px] w-3">{i}</span>
                  <div className="flex-1 flex items-center h-2.5">
                    <div
                      className="h-full rounded-sm min-w-[2px]"
                      style={{
                        width: `${Math.min(Math.abs(v), 1) * 48}px`,
                        backgroundColor: v >= 0 ? '#4ade80' : '#f87171',
                      }}
                    />
                  </div>
                  <span className="font-mono text-[11px] w-12 text-right" style={{ color }}>
                    {fmt(v)}
                  </span>
                </div>
              ))}
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
