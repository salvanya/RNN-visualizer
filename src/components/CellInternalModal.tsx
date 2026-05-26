import { useState, useEffect } from 'react';
import { X, ChevronLeft, ChevronRight } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { useStore } from '../state/store';
import { appData, getTranslationScenario } from '../data/index';
import { encoderTokenColor, decoderTokenColor } from '../utils/colors';
import { fmt } from '../utils/math-format';
import MatrixDisplay from './MatrixDisplay';
import type {
  GruLayerWeights,
  LstmLayerWeights,
  EncoderLayerStateGru,
  EncoderLayerStateLstm,
  EncoderTimestep,
  TranslationScenario,
} from '../data/types';

// ─── Helpers ─────────────────────────────────────────────────────────────────

function VecRow({ label, values, color = '#94a3b8', highlight = false }: {
  label: string;
  values: number[];
  color?: string;
  highlight?: boolean;
}) {
  return (
    <div className={`flex flex-col gap-0.5 ${highlight ? 'bg-gray-800/80 rounded-lg p-2' : ''}`}>
      <span className="font-mono text-[10px] text-gray-500">{label}</span>
      <div className="flex flex-wrap gap-x-2 gap-y-0.5">
        {values.map((v, i) => (
          <span key={i} className="font-mono text-[11px]" style={{ color }}>
            {fmt(v)}
          </span>
        ))}
      </div>
    </div>
  );
}

// ─── GRU steps ───────────────────────────────────────────────────────────────

interface GruStep {
  title: string;
  formula: string;
  description: string;
  content: (gs: EncoderLayerStateGru, ws: GruLayerWeights) => React.ReactNode;
}

const GRU_STEPS: GruStep[] = [
  {
    title: 'Gate de reset',
    formula: String.raw`r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)`,
    description: 'Decide qué parte del estado anterior "olvidar" al calcular el candidato.',
    content: (gs, _ws) => (
      <div className="flex flex-col gap-3">
        <VecRow label="W_r · x_t" values={gs.gates.reset.W_ih_x} color="#60a5fa" />
        <VecRow label="U_r · h_{t−1}" values={gs.gates.reset.W_hh_h} color="#a78bfa" />
        <VecRow label="suma" values={gs.gates.reset.sum} color="#94a3b8" />
        <VecRow label="r_t = σ(suma)" values={gs.gates.reset.sigmoid} color="#34d399" highlight />
      </div>
    ),
  },
  {
    title: 'Gate de actualización',
    formula: String.raw`z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)`,
    description: 'Controla cuánto del estado anterior conservar vs. cuánto actualizar.',
    content: (gs, _ws) => (
      <div className="flex flex-col gap-3">
        <VecRow label="W_z · x_t" values={gs.gates.update.W_ih_x} color="#60a5fa" />
        <VecRow label="U_z · h_{t−1}" values={gs.gates.update.W_hh_h} color="#a78bfa" />
        <VecRow label="suma" values={gs.gates.update.sum} color="#94a3b8" />
        <VecRow label="z_t = σ(suma)" values={gs.gates.update.sigmoid} color="#f59e0b" highlight />
      </div>
    ),
  },
  {
    title: 'Estado candidato',
    formula: String.raw`\tilde{h}_t = \tanh(W_n x_t + U_n (r_t \odot h_{t-1}) + b_n)`,
    description: 'Nueva "propuesta" de estado, filtrada por el gate de reset.',
    content: (gs, _ws) => (
      <div className="flex flex-col gap-3">
        <VecRow label="W_n · x_t" values={gs.gates.candidate.W_ih_x} color="#60a5fa" />
        <VecRow label="U_n · (r_t ⊙ h_{t−1})" values={gs.gates.candidate.W_hh_rh} color="#a78bfa" />
        <VecRow label="suma" values={gs.gates.candidate.sum} color="#94a3b8" />
        <VecRow label="ñ_t = tanh(suma)" values={gs.gates.candidate.tanh} color="#fb923c" highlight />
      </div>
    ),
  },
  {
    title: 'Estado de salida',
    formula: String.raw`h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t`,
    description: 'Interpolación entre el estado anterior y el candidato, ponderada por z_t.',
    content: (gs, _ws) => (
      <div className="flex flex-col gap-3">
        <VecRow label="z_t (update gate)" values={gs.gates.update.sigmoid} color="#f59e0b" />
        <VecRow label="h_{t−1}" values={gs.h_prev} color="#a78bfa" />
        <VecRow label="ñ_t (candidato)" values={gs.gates.candidate.tanh} color="#fb923c" />
        <VecRow label="h_t (salida)" values={gs.h_t} color="#4ade80" highlight />
      </div>
    ),
  },
];

// ─── LSTM steps ──────────────────────────────────────────────────────────────

interface LstmStep {
  title: string;
  formula: string;
  description: string;
  content: (gs: EncoderLayerStateLstm, ws: LstmLayerWeights) => React.ReactNode;
}

const LSTM_STEPS: LstmStep[] = [
  {
    title: 'Gate de olvido',
    formula: String.raw`f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)`,
    description: 'Decide qué información del estado de celda C_{t-1} descartar.',
    content: (gs, _ws) => (
      <div className="flex flex-col gap-3">
        <VecRow label="W_f · x_t" values={gs.gates.forget.W_ih_x} color="#60a5fa" />
        <VecRow label="U_f · h_{t−1}" values={gs.gates.forget.W_hh_h} color="#a78bfa" />
        <VecRow label="suma" values={gs.gates.forget.sum} color="#94a3b8" />
        <VecRow label="f_t = σ(suma)" values={gs.gates.forget.sigmoid!} color="#f87171" highlight />
      </div>
    ),
  },
  {
    title: 'Gate de entrada',
    formula: String.raw`i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)`,
    description: 'Controla qué nuevos valores agregar al estado de celda.',
    content: (gs, _ws) => (
      <div className="flex flex-col gap-3">
        <VecRow label="W_i · x_t" values={gs.gates.input.W_ih_x} color="#60a5fa" />
        <VecRow label="U_i · h_{t−1}" values={gs.gates.input.W_hh_h} color="#a78bfa" />
        <VecRow label="suma" values={gs.gates.input.sum} color="#94a3b8" />
        <VecRow label="i_t = σ(suma)" values={gs.gates.input.sigmoid!} color="#34d399" highlight />
      </div>
    ),
  },
  {
    title: 'Celda candidata',
    formula: String.raw`\tilde{C}_t = \tanh(W_g x_t + U_g h_{t-1} + b_g)`,
    description: 'Nuevos valores candidatos para agregar al estado de celda.',
    content: (gs, _ws) => (
      <div className="flex flex-col gap-3">
        <VecRow label="W_g · x_t" values={gs.gates.candidate.W_ih_x} color="#60a5fa" />
        <VecRow label="U_g · h_{t−1}" values={gs.gates.candidate.W_hh_h} color="#a78bfa" />
        <VecRow label="suma" values={gs.gates.candidate.sum} color="#94a3b8" />
        <VecRow label="C̃_t = tanh(suma)" values={gs.gates.candidate.tanh!} color="#fb923c" highlight />
      </div>
    ),
  },
  {
    title: 'Estado de celda',
    formula: String.raw`C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t`,
    description: 'La "memoria a largo plazo": combinación ponderada de lo viejo y lo nuevo.',
    content: (gs, _ws) => (
      <div className="flex flex-col gap-3">
        <VecRow label="f_t" values={gs.gates.forget.sigmoid!} color="#f87171" />
        <VecRow label="C_{t−1}" values={gs.c_prev} color="#a78bfa" />
        <VecRow label="i_t" values={gs.gates.input.sigmoid!} color="#34d399" />
        <VecRow label="C̃_t" values={gs.gates.candidate.tanh!} color="#fb923c" />
        <VecRow label="C_t" values={gs.gates.c_t} color="#facc15" highlight />
      </div>
    ),
  },
  {
    title: 'Gate de salida',
    formula: String.raw`o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)`,
    description: 'Decide qué parte del estado de celda exponer como estado oculto h_t.',
    content: (gs, _ws) => (
      <div className="flex flex-col gap-3">
        <VecRow label="W_o · x_t" values={gs.gates.output.W_ih_x} color="#60a5fa" />
        <VecRow label="U_o · h_{t−1}" values={gs.gates.output.W_hh_h} color="#a78bfa" />
        <VecRow label="suma" values={gs.gates.output.sum} color="#94a3b8" />
        <VecRow label="o_t = σ(suma)" values={gs.gates.output.sigmoid!} color="#c084fc" highlight />
      </div>
    ),
  },
  {
    title: 'Estado oculto',
    formula: String.raw`h_t = o_t \odot \tanh(C_t)`,
    description: 'La salida de la celda: filtra el estado de celda con el gate de salida.',
    content: (gs, _ws) => (
      <div className="flex flex-col gap-3">
        <VecRow label="o_t" values={gs.gates.output.sigmoid!} color="#c084fc" />
        <VecRow label="C_t" values={gs.gates.c_t} color="#facc15" />
        <VecRow label="h_t (salida)" values={gs.h_t} color="#4ade80" highlight />
      </div>
    ),
  },
];

// ─── WeightsPanel ─────────────────────────────────────────────────────────────

function GruWeightsPanel({ weights, step }: { weights: GruLayerWeights; step: number }) {
  const panels = [
    { W: weights.W_ih_r, U: weights.W_hh_r, b: weights.b_r, wLabel: 'W_r', uLabel: 'U_r' },
    { W: weights.W_ih_z, U: weights.W_hh_z, b: weights.b_z, wLabel: 'W_z', uLabel: 'U_z' },
    { W: weights.W_ih_n, U: weights.W_hh_n, b: weights.b_n, wLabel: 'W_n', uLabel: 'U_n' },
    null, // step 3: output formula, no new weights
  ];
  const p = step < 3 ? panels[step] : null;
  if (!p) return (
    <div className="text-gray-600 text-[10px] font-mono italic text-center pt-4">
      Combinación con pesos ya calculados
    </div>
  );
  return (
    <div className="flex flex-col gap-4">
      <MatrixDisplay matrix={p.W} label={`${p.wLabel} (dim_capa × dim_entrada)`} />
      <MatrixDisplay matrix={p.U} label={`${p.uLabel} (dim_capa × dim_capa)`} />
      <div className="flex flex-col gap-0.5">
        <span className="font-mono text-[10px] text-gray-400">bias b</span>
        <div className="flex gap-1.5 flex-wrap">
          {p.b.map((v, i) => (
            <span key={i} className="font-mono text-[10px] text-gray-300">{v.toFixed(2)}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

function LstmWeightsPanel({ weights, step }: { weights: LstmLayerWeights; step: number }) {
  const panels = [
    { W: weights.W_ih_f, U: weights.W_hh_f, b: weights.b_f, wLabel: 'W_f', uLabel: 'U_f' },
    { W: weights.W_ih_i, U: weights.W_hh_i, b: weights.b_i, wLabel: 'W_i', uLabel: 'U_i' },
    { W: weights.W_ih_g, U: weights.W_hh_g, b: weights.b_g, wLabel: 'W_g', uLabel: 'U_g' },
    null,
    { W: weights.W_ih_o, U: weights.W_hh_o, b: weights.b_o, wLabel: 'W_o', uLabel: 'U_o' },
    null,
  ];
  const p = panels[step];
  if (!p) return (
    <div className="text-gray-600 text-[10px] font-mono italic text-center pt-4">
      Combinación con pesos ya calculados
    </div>
  );
  return (
    <div className="flex flex-col gap-4">
      <MatrixDisplay matrix={p.W} label={`${p.wLabel} (dim_capa × dim_entrada)`} />
      <MatrixDisplay matrix={p.U} label={`${p.uLabel} (dim_capa × dim_capa)`} />
      <div className="flex flex-col gap-0.5">
        <span className="font-mono text-[10px] text-gray-400">bias b</span>
        <div className="flex gap-1.5 flex-wrap">
          {p.b.map((v, i) => (
            <span key={i} className="font-mono text-[10px] text-gray-300">{v.toFixed(2)}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Main modal ──────────────────────────────────────────────────────────────

export default function CellInternalModal() {
  const { modalCeldaAbierta, cerrarModalCelda, arquitectura, modo, atencion } = useStore();
  const [step, setStep] = useState(0);

  const isGru = arquitectura === 'GRU';
  const totalSteps = isGru ? 4 : 6;

  // Reset step when modal opens
  useEffect(() => {
    if (modalCeldaAbierta) setStep(0);
  }, [modalCeldaAbierta]);

  // Esc key to close
  useEffect(() => {
    if (!modalCeldaAbierta) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') cerrarModalCelda();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [modalCeldaAbierta, cerrarModalCelda]);

  if (!modalCeldaAbierta) return null;

  const { layer, t, lado } = modalCeldaAbierta;
  const isDecoder = lado === 'dec';

  // Resolve scenario and extract layer state + weights
  let layerState: EncoderLayerStateGru | EncoderLayerStateLstm;
  let weights: GruLayerWeights | LstmLayerWeights;
  let token: string;
  let color: string;
  let layerDim: string;
  let inputDim: string;

  if (!isDecoder) {
    const attnSuffix =
      atencion === "none" ? "noattn" : atencion === "bahdanau" ? "attn_bahdanau" : "attn";
    const scenarioKey = modo === 'sentiment'
      ? `${arquitectura}_sentiment`
      : `${arquitectura}_translation_${attnSuffix}`;
    const scenarios = appData.scenarios as unknown as Record<string, {
      encoder: { weights: { layer1: GruLayerWeights | LstmLayerWeights; layer2: GruLayerWeights | LstmLayerWeights }; timesteps: EncoderTimestep[] }
    }>;
    const scenario = scenarios[scenarioKey];
    const tsData = scenario.encoder.timesteps[t - 1];
    layerState = layer === 1 ? tsData.layer1 : tsData.layer2;
    weights = scenario.encoder.weights[`layer${layer}` as 'layer1' | 'layer2'];
    token = tsData.input;
    color = encoderTokenColor(token);
    layerDim = layer === 1 ? 'm' : 'l';
    inputDim = layer === 1 ? 'd' : 'm';
  } else {
    const translationScenario = getTranslationScenario(arquitectura, atencion);
    const tsData = translationScenario.decoder.timesteps[t - 1];
    layerState = (layer === 1 ? tsData.layer1 : tsData.layer2) as EncoderLayerStateGru | EncoderLayerStateLstm;
    weights = (translationScenario as unknown as TranslationScenario & {
      decoder: { weights: { layer1: GruLayerWeights | LstmLayerWeights; layer2: GruLayerWeights | LstmLayerWeights } }
    }).decoder.weights[`layer${layer}` as 'layer1' | 'layer2'];
    token = tsData.input_token;
    color = decoderTokenColor(tsData.softmax.argmax);
    layerDim = layer === 1 ? 'p' : 'q';
    inputDim = layer === 1 ? 'd' : 'p';
  }

  const isLstm = 'c_prev' in layerState;
  const steps = isGru ? GRU_STEPS : LSTM_STEPS;
  const currentStep = steps[step];

  return (
    <AnimatePresence>
      <motion.div
        key="overlay"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.15 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/70"
        onClick={(e) => { if (e.target === e.currentTarget) cerrarModalCelda(); }}
      >
        <motion.div
          key="modal"
          initial={{ opacity: 0, scale: 0.95, y: 8 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 8 }}
          transition={{ duration: 0.2, ease: 'easeOut' }}
          className="bg-gray-900 border rounded-2xl shadow-2xl w-[90vw] max-w-5xl max-h-[85vh] flex flex-col overflow-hidden"
          style={{ borderColor: `${color}44` }}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800 shrink-0">
            <div className="flex items-center gap-3">
              <div
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: color, boxShadow: `0 0 8px ${color}` }}
              />
              <span className="font-mono text-sm font-semibold" style={{ color }}>
                Celda {arquitectura} — {isDecoder ? 'Dec' : 'Enc'} Capa {layer}, t={t} ("{token}")
              </span>
              <span className="text-gray-600 text-xs">
                dim entrada={inputDim}, dim capa={layerDim}
              </span>
            </div>
            <button
              onClick={cerrarModalCelda}
              className="text-gray-500 hover:text-gray-200 transition-colors"
            >
              <X size={18} />
            </button>
          </div>

          {/* Body */}
          <div className="flex flex-1 overflow-hidden">
            {/* Left column: inputs */}
            <div className="w-48 shrink-0 border-r border-gray-800 p-4 flex flex-col gap-4 overflow-y-auto">
              <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">
                Entradas
              </div>
              <VecRow
                label={isDecoder ? `emb("${token}") (dim ${inputDim})` : `x_${t} (embedding, dim ${inputDim})`}
                values={layerState.x_t}
                color="#94a3b8"
              />
              <VecRow
                label={`h_{${t - 1}}^(${layer}) (dim ${layerDim})`}
                values={layerState.h_prev}
                color={`${color}bb`}
              />
              {isLstm && (
                <VecRow
                  label={`C_{${t - 1}}^(${layer}) (dim ${layerDim})`}
                  values={(layerState as EncoderLayerStateLstm).c_prev}
                  color="#facc1599"
                />
              )}
            </div>

            {/* Center: step navigator */}
            <div className="flex-1 flex flex-col overflow-hidden">
              {/* Step header */}
              <div className="px-6 py-3 border-b border-gray-800 shrink-0">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-semibold text-gray-300">
                    Paso {step + 1} / {totalSteps} — {currentStep.title}
                  </span>
                  <div className="flex gap-1">
                    {steps.map((_, i) => (
                      <button
                        key={i}
                        onClick={() => setStep(i)}
                        className="w-5 h-5 rounded-full text-[9px] font-mono transition-colors"
                        style={{
                          backgroundColor: i === step ? color : '#374151',
                          color: i === step ? '#000' : '#9ca3af',
                        }}
                      >
                        {i + 1}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="font-mono text-[11px] text-gray-400">{currentStep.description}</div>
              </div>

              {/* Formula */}
              <div className="px-6 py-3 border-b border-gray-800 bg-gray-950/50 shrink-0">
                <div className="text-[10px] text-gray-500 mb-1.5 uppercase tracking-wider">Fórmula</div>
                <div className="text-white text-sm">
                  <InlineMath math={currentStep.formula} />
                </div>
              </div>

              {/* Step content */}
              <div className="flex-1 overflow-y-auto p-6">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={step}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    transition={{ duration: 0.18 }}
                  >
                    {isGru
                      ? (GRU_STEPS[step] as GruStep).content(
                          layerState as EncoderLayerStateGru,
                          weights as GruLayerWeights
                        )
                      : (LSTM_STEPS[step] as LstmStep).content(
                          layerState as EncoderLayerStateLstm,
                          weights as LstmLayerWeights
                        )
                    }
                  </motion.div>
                </AnimatePresence>
              </div>

              {/* Navigation footer */}
              <div className="px-6 py-3 border-t border-gray-800 shrink-0 flex items-center justify-between">
                <button
                  onClick={() => setStep((s) => Math.max(0, s - 1))}
                  disabled={step === 0}
                  className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-200 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronLeft size={14} /> Anterior
                </button>
                <span className="text-[10px] text-gray-600 font-mono">
                  {step + 1} / {totalSteps}
                </span>
                <button
                  onClick={() => setStep((s) => Math.min(totalSteps - 1, s + 1))}
                  disabled={step === totalSteps - 1}
                  className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-200 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  Siguiente <ChevronRight size={14} />
                </button>
              </div>
            </div>

            {/* Right column: weights + output */}
            <div className="w-52 shrink-0 border-l border-gray-800 flex flex-col overflow-hidden">
              {/* Weights */}
              <div className="flex-1 overflow-y-auto p-4 border-b border-gray-800">
                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold mb-3">
                  Matrices (paso {step + 1})
                </div>
                <AnimatePresence mode="wait">
                  <motion.div
                    key={step}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.15 }}
                  >
                    {isGru
                      ? <GruWeightsPanel weights={weights as GruLayerWeights} step={step} />
                      : <LstmWeightsPanel weights={weights as LstmLayerWeights} step={step} />
                    }
                  </motion.div>
                </AnimatePresence>
              </div>

              {/* Output */}
              <div className="p-4 shrink-0">
                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold mb-3">
                  Salida
                </div>
                <VecRow
                  label={`h_${t}^(${layer}) (dim ${layerDim})`}
                  values={layerState.h_t}
                  color={color}
                  highlight
                />
                {isLstm && (
                  <div className="mt-3">
                    <VecRow
                      label={`C_${t}^(${layer}) (dim ${layerDim})`}
                      values={(layerState as EncoderLayerStateLstm).c_t}
                      color="#facc15"
                      highlight
                    />
                  </div>
                )}
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
