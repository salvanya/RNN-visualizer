import { motion, AnimatePresence } from 'framer-motion';
import { useStore } from '../state/store';
import { getDecoderTimesteps } from '../data/index';
import { decoderTokenColor } from '../utils/colors';
import { durationSec } from '../utils/timing';

export default function OutputSentence() {
  const { timestep, arquitectura, atencion, velocidad } = useStore();
  const decTimesteps = getDecoderTimesteps(arquitectura, atencion);

  const decStep = Math.min(Math.max(timestep - 7, 0), 6);
  const tokens = decTimesteps.slice(0, decStep).map((ts) => ts.softmax.argmax);

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-[11px] text-gray-500 font-mono shrink-0">Salida (EN):</span>
      {tokens.length === 0 ? (
        <span className="text-[11px] text-gray-700 font-mono italic">…esperando decoder</span>
      ) : (
        tokens.map((tok, i) => (
          <AnimatePresence key={i}>
            <motion.span
              initial={{ opacity: 0, y: -6, scale: 0.8 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: durationSec('embeddingAparece', velocidad) }}
              className="font-mono text-sm font-semibold px-2 py-0.5 rounded-md"
              style={{
                color: decoderTokenColor(tok),
                backgroundColor: `${decoderTokenColor(tok)}18`,
                border: `1px solid ${decoderTokenColor(tok)}44`,
              }}
            >
              {tok}
            </motion.span>
          </AnimatePresence>
        ))
      )}
    </div>
  );
}
