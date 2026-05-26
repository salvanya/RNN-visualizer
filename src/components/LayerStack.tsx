import { Microscope } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useStore } from '../state/store';
import { durationSec } from '../utils/timing';
import LayerFrame from './LayerFrame';
import UnitNode, { type UnitTooltipRow } from './UnitNode';
import type {
  EncoderLayerState,
  EncoderLayerStateLstm,
  GruLayerWeights,
  LstmLayerWeights,
} from '../data/types';
import { isLstmLayer } from '../data/types';

// ─── Layout constants ──────────────────────────────────────────────────────

const LEFT_GUTTER = 56;
const COL_PITCH_BASE = 156;
const TOP_PADDING = 26;
const HEADER_HEIGHT = 22;
const HEADER_TO_INPUTS = 16;
const INPUT_NODE_SIZE = 32;
const INPUT_NODE_PITCH_X = 38;   // inputs fanean horizontalmente — pitch > size
const INPUT_TO_UNITS_GAP = 64;
const UNIT_NODE_SIZE = 46;
const UNIT_NODE_PITCH = 56;
const FRAME_BOTTOM_PADDING = 22;
const LAYER_GAP = 22;
const L2_HEADER_OFFSET = 16;
const L2_FANIN_GAP = 56;

// ─── Weight aggregation ───────────────────────────────────────────────────

type AnyLayerWeights = GruLayerWeights | LstmLayerWeights;

function gateKeys(isLstm: boolean): string[] {
  return isLstm ? ['f', 'i', 'g', 'o'] : ['r', 'z', 'n'];
}

function aggregateWeight(
  weights: AnyLayerWeights,
  type: 'hh' | 'ih',
  i: number,
  j: number,
  isLstm: boolean
): number {
  let maxAbs = 0;
  let dominant = 0;
  for (const gk of gateKeys(isLstm)) {
    const key = `W_${type}_${gk}` as keyof AnyLayerWeights;
    const m = weights[key] as number[][];
    if (!m || !m[i]) continue;
    const w = m[i][j];
    if (Math.abs(w) > maxAbs) {
      maxAbs = Math.abs(w);
      dominant = w;
    }
  }
  return dominant;
}

function edgeStyle(w: number): { stroke: string; width: number; opacity: number } {
  const a = Math.abs(w);
  if (a < 0.04) return { stroke: '#475569', width: 0.4, opacity: 0.15 };
  const stroke = w > 0 ? '#22c55e' : '#ef4444';
  return {
    stroke,
    width: Math.max(0.5, Math.min(2.6, 0.4 + a * 1.6)),
    opacity: Math.max(0.22, Math.min(0.85, 0.28 + a * 0.55)),
  };
}

// ─── Types ────────────────────────────────────────────────────────────────

export interface LayerStackTimestep {
  t: number;                  // 1..N
  tokenLabel: string;         // 'encantó' (encoder) o token predicho (decoder)
  inputTokenLabel?: string;   // para decoder: token de entrada (puede diferir del predicho)
  tokenColor: string;
  inputVector: number[];      // embedding (dim d) o h^(1) según capa — para LAYER 1 inputs
  layer1: EncoderLayerState;
  layer2: EncoderLayerState;
}

interface Props {
  side: 'enc' | 'dec';
  accentColor: string;         // azul para enc, naranja para dec
  layer1Units: number;         // m o p
  layer2Units: number;         // l o q
  inputDimL1: number;          // d (embedding dim) o d+l (Bahdanau con contexto concatenado)
  // Si está presente, los primeros `embeddingDim` inputs son embedding (color del token)
  // y los siguientes (inputDimL1 - embeddingDim) son contexto de atención (color cálido).
  embeddingDim?: number;
  visibleCount: number;
  activeT: number | null;
  timesteps: LayerStackTimestep[];
  weightsL1: AnyLayerWeights;
  weightsL2: AnyLayerWeights;
  isLstm: boolean;
}

// ─── Component ────────────────────────────────────────────────────────────

export default function LayerStack({
  side,
  accentColor,
  layer1Units,
  layer2Units,
  inputDimL1,
  embeddingDim,
  visibleCount,
  activeT,
  timesteps,
  weightsL1,
  weightsL2,
  isLstm,
}: Props) {
  const { abrirModalCelda, velocidad } = useStore();

  // ── Layout computation ──
  // Pitch dinámico: si inputDimL1 es grande (Bahdanau con dim 8), la columna
  // necesita más ancho para que los inputs faneados no se superpongan con la
  // columna vecina ni se tapen los tokens.
  const inputSpan = (inputDimL1 - 1) * INPUT_NODE_PITCH_X;
  const COL_PITCH = Math.max(COL_PITCH_BASE, inputSpan + 56);
  const colX = (tIdx: number) => LEFT_GUTTER + (tIdx + 0.5) * COL_PITCH;
  // Inputs FANEAN HORIZONTALMENTE: x relativo al centro de la columna
  const inputOffsetX = (k: number) => (k - (inputDimL1 - 1) / 2) * INPUT_NODE_PITCH_X;
  const l1InputX = (tIdx: number, k: number) => colX(tIdx) + inputOffsetX(k);

  // Layer 1 Y-coords
  const L1_HEADER_Y = TOP_PADDING;
  const L1_INPUT_Y = L1_HEADER_Y + HEADER_HEIGHT + HEADER_TO_INPUTS + INPUT_NODE_SIZE / 2;
  const L1_INPUT_BOTTOM = L1_INPUT_Y + INPUT_NODE_SIZE / 2;
  const L1_UNITS_TOP_CENTER = L1_INPUT_BOTTOM + INPUT_TO_UNITS_GAP + UNIT_NODE_SIZE / 2;
  const l1UnitY = (i: number) => L1_UNITS_TOP_CENTER + i * UNIT_NODE_PITCH;
  const L1_UNITS_BOTTOM = l1UnitY(layer1Units - 1) + UNIT_NODE_SIZE / 2;
  const L1_FRAME_HEIGHT = L1_UNITS_BOTTOM + FRAME_BOTTOM_PADDING;

  // Layer 2 Y-coords
  const L2_FRAME_TOP = L1_FRAME_HEIGHT + LAYER_GAP;
  const L2_HEADER_Y = L2_FRAME_TOP + L2_HEADER_OFFSET;
  const L2_UNITS_TOP_CENTER = L2_HEADER_Y + HEADER_HEIGHT + L2_FANIN_GAP + UNIT_NODE_SIZE / 2;
  const l2UnitY = (i: number) => L2_UNITS_TOP_CENTER + i * UNIT_NODE_PITCH;
  const L2_UNITS_BOTTOM = l2UnitY(layer2Units - 1) + UNIT_NODE_SIZE / 2;
  const L2_FRAME_HEIGHT = L2_UNITS_BOTTOM + FRAME_BOTTOM_PADDING - L2_FRAME_TOP;

  const TOTAL_HEIGHT = L2_FRAME_TOP + L2_FRAME_HEIGHT;
  const TOTAL_WIDTH = LEFT_GUTTER + Math.max(visibleCount, 1) * COL_PITCH + 24;

  // ── Edge geometry helpers ──
  const halfNode = UNIT_NODE_SIZE / 2;

  // Línea recta entre dos puntos (cuando comparten X o Y, o cuando da igual la forma)
  function straightPath(x1: number, y1: number, x2: number, y2: number): string {
    return `M ${x1} ${y1} L ${x2} ${y2}`;
  }
  // Bezier S-shape horizontal (W_hh entre timesteps)
  function bezierH(x1: number, y1: number, x2: number, y2: number): string {
    const mx = (x1 + x2) / 2;
    return `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
  }
  // Bezier S-shape vertical CON BOW lateral (W_ih dentro de una columna, cuando inicio y fin comparten X)
  function bezierVBowed(x1: number, y1: number, x2: number, y2: number, bow: number): string {
    const my = (y1 + y2) / 2;
    const cx1 = x1 + bow;
    const cx2 = x2 + bow;
    return `M ${x1} ${y1} C ${cx1} ${my}, ${cx2} ${my}, ${x2} ${y2}`;
  }
  // Bezier diagonal genérica (input fanned X1 → unit X2)
  function bezierDiag(x1: number, y1: number, x2: number, y2: number): string {
    const my = (y1 + y2) / 2;
    return `M ${x1} ${y1} C ${x1} ${my}, ${x2} ${my}, ${x2} ${y2}`;
  }

  // ── Render edges ──
  const edges: React.ReactNode[] = [];

  for (let tIdx = 0; tIdx < visibleCount; tIdx++) {
    const cx = colX(tIdx);

    // — W_ih layer 1: input[k] (X = l1InputX) → l1.unit[i] (X = cx, top)
    // Inputs fanean horizontalmente, edges diagonales naturales
    for (let i = 0; i < layer1Units; i++) {
      for (let k = 0; k < inputDimL1; k++) {
        const w = aggregateWeight(weightsL1, 'ih', i, k, isLstm);
        const s = edgeStyle(w);
        const x1 = l1InputX(tIdx, k);
        const y1 = L1_INPUT_BOTTOM;
        const x2 = cx;
        const y2 = l1UnitY(i) - halfNode;
        edges.push(
          <path
            key={`ih1-${tIdx}-${i}-${k}`}
            d={bezierDiag(x1, y1, x2, y2)}
            stroke={s.stroke}
            strokeWidth={s.width}
            strokeOpacity={s.opacity}
            fill="none"
          />
        );
      }
    }

    // — W_ih layer 2: l1.unit[k] → l2.unit[i] (cross-frame), todos comparten X=cx
    // → uso bezier con BOW lateral basado en (k,i) para que se separen visualmente
    const l1Mid = (layer1Units - 1) / 2;
    const l2Mid = (layer2Units - 1) / 2;
    for (let i = 0; i < layer2Units; i++) {
      for (let k = 0; k < layer1Units; k++) {
        const w = aggregateWeight(weightsL2, 'ih', i, k, isLstm);
        const s = edgeStyle(w);
        const y1 = l1UnitY(k) + halfNode;
        const y2 = l2UnitY(i) - halfNode;
        const bow = (k - l1Mid) * 8 + (i - l2Mid) * 6;
        edges.push(
          <path
            key={`ih2-${tIdx}-${i}-${k}`}
            d={bezierVBowed(cx, y1, cx, y2, bow)}
            stroke={s.stroke}
            strokeWidth={s.width}
            strokeOpacity={s.opacity}
            fill="none"
          />
        );
      }
    }

    // — W_hh entre timesteps (horizontal, todas-a-todas)
    // En LSTM, h en sección SUPERIOR de la celda, C en sección INFERIOR.
    // → W_hh ataca a la sección superior; C edges atacan a la sección inferior.
    const hOffset = isLstm ? -halfNode * 0.45 : 0;
    const cOffset = halfNode * 0.45;

    if (tIdx > 0) {
      const cxPrev = colX(tIdx - 1);
      const startX = cxPrev + halfNode;
      const endX = cx - halfNode;

      // W_hh layer 1
      for (let i = 0; i < layer1Units; i++) {
        for (let j = 0; j < layer1Units; j++) {
          const w = aggregateWeight(weightsL1, 'hh', i, j, isLstm);
          const s = edgeStyle(w);
          edges.push(
            <path
              key={`hh1-${tIdx}-${i}-${j}`}
              d={bezierH(startX, l1UnitY(j) + hOffset, endX, l1UnitY(i) + hOffset)}
              stroke={s.stroke}
              strokeWidth={s.width}
              strokeOpacity={s.opacity}
              fill="none"
            />
          );
        }
      }
      // C edges layer 1 (LSTM) — entran/salen por la sección inferior de la celda
      if (isLstm) {
        for (let i = 0; i < layer1Units; i++) {
          const cy = l1UnitY(i) + cOffset;
          edges.push(
            <path
              key={`c1-${tIdx}-${i}`}
              d={straightPath(startX, cy, endX, cy)}
              stroke="#facc15"
              strokeWidth={1.6}
              strokeOpacity={0.8}
              strokeDasharray="4 2"
              fill="none"
            />
          );
        }
      }

      // W_hh layer 2
      for (let i = 0; i < layer2Units; i++) {
        for (let j = 0; j < layer2Units; j++) {
          const w = aggregateWeight(weightsL2, 'hh', i, j, isLstm);
          const s = edgeStyle(w);
          edges.push(
            <path
              key={`hh2-${tIdx}-${i}-${j}`}
              d={bezierH(startX, l2UnitY(j) + hOffset, endX, l2UnitY(i) + hOffset)}
              stroke={s.stroke}
              strokeWidth={s.width}
              strokeOpacity={s.opacity}
              fill="none"
            />
          );
        }
      }
      if (isLstm) {
        for (let i = 0; i < layer2Units; i++) {
          const cy = l2UnitY(i) + cOffset;
          edges.push(
            <path
              key={`c2-${tIdx}-${i}`}
              d={straightPath(startX, cy, endX, cy)}
              stroke="#facc15"
              strokeWidth={1.6}
              strokeOpacity={0.8}
              strokeDasharray="4 2"
              fill="none"
            />
          );
        }
      }
    }
  }

  // ── Empty state ──
  if (visibleCount === 0) {
    return (
      <div className="relative" style={{ width: TOTAL_WIDTH, height: TOTAL_HEIGHT }}>
        <LayerFrame
          title={`Capa 1 · ${layer1Units} unidades`}
          subtitle={isLstm ? 'h, C inicializados en 0' : 'h inicializado en 0'}
          color={accentColor}
          top={0}
          height={L1_FRAME_HEIGHT}
          width={TOTAL_WIDTH}
        />
        <LayerFrame
          title={`Capa 2 · ${layer2Units} unidades`}
          subtitle={isLstm ? 'h, C inicializados en 0' : 'h inicializado en 0'}
          color={accentColor}
          top={L2_FRAME_TOP}
          height={L2_FRAME_HEIGHT}
          width={TOTAL_WIDTH}
        />
      </div>
    );
  }

  // ── Render ──
  return (
    <div className="relative" style={{ width: TOTAL_WIDTH, height: TOTAL_HEIGHT }}>
      {/* Frames (background) */}
      <LayerFrame
        title={`Capa 1 · ${layer1Units} unidades`}
        subtitle={isLstm ? `LSTM · estados h, C (dim ${layer1Units} cada uno)` : `GRU · estado oculto h (dim ${layer1Units})`}
        color={accentColor}
        top={0}
        height={L1_FRAME_HEIGHT}
        width={TOTAL_WIDTH}
      />
      <LayerFrame
        title={`Capa 2 · ${layer2Units} unidades`}
        subtitle={isLstm ? `LSTM · estados h, C (dim ${layer2Units} cada uno)` : `GRU · estado oculto h (dim ${layer2Units})`}
        color={accentColor}
        top={L2_FRAME_TOP}
        height={L2_FRAME_HEIGHT}
        width={TOTAL_WIDTH}
      />

      {/* SVG edges */}
      <svg
        className="absolute inset-0 pointer-events-none"
        width={TOTAL_WIDTH}
        height={TOTAL_HEIGHT}
        style={{ overflow: 'visible' }}
      >
        {edges}
      </svg>

      {/* Left gutter labels — unit row indices */}
      <div className="absolute" style={{ left: 8, top: L1_UNITS_TOP_CENTER - 8 }}>
        {Array.from({ length: layer1Units }, (_, i) => (
          <div
            key={i}
            className="font-mono text-[9px] text-gray-500"
            style={{ position: 'absolute', top: i * UNIT_NODE_PITCH - 4, left: 0 }}
          >
            [{i + 1}]
          </div>
        ))}
      </div>
      <div className="absolute" style={{ left: 8, top: L2_UNITS_TOP_CENTER - 8 }}>
        {Array.from({ length: layer2Units }, (_, i) => (
          <div
            key={i}
            className="font-mono text-[9px] text-gray-500"
            style={{ position: 'absolute', top: i * UNIT_NODE_PITCH - 4, left: 0 }}
          >
            [{i + 1}]
          </div>
        ))}
      </div>

      {/* Per-column content */}
      <AnimatePresence>
        {Array.from({ length: visibleCount }, (_, tIdx) => {
          const ts = timesteps[tIdx];
          const cx = colX(tIdx);
          const isActiveCol = activeT === tIdx + 1;
          const colColor = ts.tokenColor;

          return (
            <motion.div
              key={`col-${tIdx}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: durationSec('desdoblamientoTimestep', velocidad) }}
            >
              {/* Active column highlight band */}
              {isActiveCol && (
                <div
                  className="absolute rounded-xl pointer-events-none"
                  style={{
                    left: cx - COL_PITCH / 2 + 8,
                    top: 4,
                    width: COL_PITCH - 16,
                    height: TOTAL_HEIGHT - 8,
                    background: `${colColor}10`,
                    border: `1px solid ${colColor}33`,
                  }}
                />
              )}

              {/* Header — Layer 1: timestep label + ver-interior button */}
              <ColumnHeader
                cx={cx}
                top={L1_HEADER_Y}
                tLabel={tIdx + 1}
                side={side}
                tokenLabel={ts.tokenLabel}
                tokenColor={colColor}
                onMicroscope={() => abrirModalCelda({ layer: 1, t: tIdx + 1, lado: side })}
                colPitch={COL_PITCH}
              />

              {/* Input nodes (layer 1) — embedding (y opcionalmente contexto de atención), fanned horizontally */}
              {ts.inputVector.map((v, k) => {
                const isContext = embeddingDim !== undefined && k >= embeddingDim;
                const nodeColor = isContext ? '#fbbf24' : colColor;
                const rowLabel = isContext ? 'contexto c_t' : 'embedding';
                const ctxIdx = isContext ? k - (embeddingDim ?? 0) : k;
                const labelText = isContext
                  ? `c_${tIdx + 1}[${ctxIdx + 1}]`
                  : `x_${tIdx + 1}[${k + 1}]`;
                return (
                  <UnitNode
                    key={`in-${tIdx}-${k}`}
                    id={`${side}-in-l1-t${tIdx + 1}-k${k}`}
                    cx={l1InputX(tIdx, k)}
                    cy={L1_INPUT_Y}
                    size={INPUT_NODE_SIZE}
                    value={v}
                    borderColor={nodeColor}
                    fillOpacity={0.08}
                    isActive={isActiveCol}
                    label={labelText}
                    tokenLabel={ts.inputTokenLabel ?? ts.tokenLabel}
                    tooltipRows={[
                      { label: rowLabel, value: v, color: nodeColor, highlight: true },
                    ]}
                    tooltipSide="above"
                  />
                );
              })}

              {/* Layer 1 unit nodes */}
              {Array.from({ length: layer1Units }, (_, i) => {
                const ls = ts.layer1;
                const isLstmL = isLstmLayer(ls);
                const hv = ls.h_t[i];
                const cv = isLstmL ? (ls as EncoderLayerStateLstm).c_t[i] : undefined;
                const hPrev = ls.h_prev[i];
                const cPrev = isLstmL ? (ls as EncoderLayerStateLstm).c_prev[i] : undefined;
                return (
                  <UnitNode
                    key={`u1-${tIdx}-${i}`}
                    id={`${side}-l1-t${tIdx + 1}-u${i}`}
                    cx={cx}
                    cy={l1UnitY(i)}
                    size={UNIT_NODE_SIZE}
                    value={hv}
                    cValue={cv}
                    borderColor={colColor}
                    fillOpacity={isActiveCol ? 0.22 : 0.13}
                    isActive={isActiveCol}
                    label={`h_${tIdx + 1}⁽¹⁾[${i + 1}]`}
                    tokenLabel={ts.tokenLabel}
                    tooltipRows={buildHRows(hv, hPrev, cv, cPrev, colColor, tIdx + 1, 1, i + 1)}
                    tooltipSide="above"
                  />
                );
              })}

              {/* Header — Layer 2 */}
              <ColumnHeader
                cx={cx}
                top={L2_HEADER_Y}
                tLabel={tIdx + 1}
                side={side}
                tokenLabel={ts.tokenLabel}
                tokenColor={colColor}
                onMicroscope={() => abrirModalCelda({ layer: 2, t: tIdx + 1, lado: side })}
                compact
                colPitch={COL_PITCH}
              />

              {/* Layer 2 unit nodes */}
              {Array.from({ length: layer2Units }, (_, i) => {
                const ls = ts.layer2;
                const isLstmL = isLstmLayer(ls);
                const hv = ls.h_t[i];
                const cv = isLstmL ? (ls as EncoderLayerStateLstm).c_t[i] : undefined;
                const hPrev = ls.h_prev[i];
                const cPrev = isLstmL ? (ls as EncoderLayerStateLstm).c_prev[i] : undefined;
                return (
                  <UnitNode
                    key={`u2-${tIdx}-${i}`}
                    id={`${side}-l2-t${tIdx + 1}-u${i}`}
                    cx={cx}
                    cy={l2UnitY(i)}
                    size={UNIT_NODE_SIZE}
                    value={hv}
                    cValue={cv}
                    borderColor={colColor}
                    fillOpacity={isActiveCol ? 0.22 : 0.13}
                    isActive={isActiveCol}
                    label={`h_${tIdx + 1}⁽²⁾[${i + 1}]`}
                    tokenLabel={ts.tokenLabel}
                    tooltipRows={buildHRows(hv, hPrev, cv, cPrev, colColor, tIdx + 1, 2, i + 1)}
                    tooltipSide="below"
                  />
                );
              })}
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}

// ─── Helpers ──────────────────────────────────────────────────────────────

function buildHRows(
  h: number,
  hPrev: number,
  c: number | undefined,
  cPrev: number | undefined,
  color: string,
  t: number,
  layer: 1 | 2,
  unit: number
): UnitTooltipRow[] {
  const sup = layer === 1 ? '⁽¹⁾' : '⁽²⁾';
  const idx = `[${unit}]`;
  const rows: UnitTooltipRow[] = [
    { label: `h_${t - 1}${sup}${idx}`, value: hPrev, color: `${color}99` },
    { label: `h_${t}${sup}${idx}`, value: h, color, highlight: true },
  ];
  if (c !== undefined && cPrev !== undefined) {
    rows.push({ label: `C_${t - 1}${sup}${idx}`, value: cPrev, color: '#ca8a04' });
    rows.push({ label: `C_${t}${sup}${idx}`, value: c, color: '#facc15', highlight: true });
  }
  return rows;
}

interface ColumnHeaderProps {
  cx: number;
  top: number;
  tLabel: number;
  side: 'enc' | 'dec';
  tokenLabel: string;
  tokenColor: string;
  onMicroscope: () => void;
  compact?: boolean;
  colPitch: number;
}

function ColumnHeader({ cx, top, tLabel, side, tokenLabel, tokenColor, onMicroscope, compact, colPitch }: ColumnHeaderProps) {
  const sup = side === 'enc' ? '⁽ᵉ⁾' : '⁽ᵈ⁾';
  return (
    <div
      className="absolute flex items-center justify-center gap-1.5 rounded-md"
      style={{
        left: cx - colPitch / 2,
        top,
        width: colPitch,
        height: HEADER_HEIGHT,
        background: '#0a0a0acc',
        zIndex: 10,
      }}
    >
      <span className="font-mono text-[10px]" style={{ color: tokenColor }}>
        t_{tLabel}{sup}
      </span>
      {!compact && (
        <span className="font-mono text-[9px] text-gray-500 truncate max-w-[60px]">
          "{tokenLabel}"
        </span>
      )}
      <button
        className="text-gray-500 hover:text-gray-200 transition-colors flex-shrink-0"
        onClick={onMicroscope}
        title={`Ver interior · capa ${compact ? 2 : 1} · t=${tLabel}`}
      >
        <Microscope size={11} />
      </button>
    </div>
  );
}
