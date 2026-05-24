import { motion, AnimatePresence } from "framer-motion";
import { useStore } from "../state/store";
import { appData } from "../data/index";
import VectorDisplay from "./VectorDisplay";
import MatrixDisplay from "./MatrixDisplay";
import { fmtPct, fmt } from "../utils/math-format";
import { durationSec } from "../utils/timing";

const CLASS_LABELS = ["Negativo", "Neutro", "Positivo"];
const CLASS_COLORS = ["#f87171", "#94a3b8", "#4ade80"];
const CLASS_BG = ["rgba(127,29,29,0.5)", "rgba(30,41,59,0.5)", "rgba(20,83,45,0.5)"];

export default function SentimentHead() {
  const { timestep, arquitectura, velocidad } = useStore();

  const scenarioKey = `${arquitectura}_sentiment` as const;
  const head = appData.scenarios[scenarioKey].head;

  const visible = timestep >= 7;
  const winnerIdx = head.softmax.indexOf(Math.max(...head.softmax));

  return (
    <AnimatePresence mode="wait">
      {!visible ? (
        <motion.div
          key="waiting"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: durationSec('cambioTooltip', velocidad) }}
          className="text-gray-600 text-xs font-mono italic px-1"
        >
          {timestep === 0
            ? "Esperando inicio del encoder…"
            : `Procesando encoder (t=${timestep}/7)…`}
        </motion.div>
      ) : (
        <motion.div
          key="head"
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: durationSec('embeddingAparece', velocidad) }}
          className="overflow-x-auto pb-2"
        >
          <div className="flex flex-row gap-5 items-start min-w-max">

            {/* ── Input: h_T^(2) ── */}
            <Section title={`h₇⁽²⁾`} subtitle="dim 4" color="#818cf8">
              <VectorDisplay values={head.input} color="#818cf8" withBars />
            </Section>

            <Arrow />

            {/* ── Dense 1 ── */}
            <Section title="Dense 1" subtitle="(8 × 4)  →  ReLU" color="#60a5fa">
              <div className="flex flex-row gap-3 items-start">
                <MatrixDisplay matrix={head.dense1.W} label="W_d1" />
                <div className="flex flex-col gap-2 pt-3">
                  <div>
                    <div className="text-[9px] text-gray-500 font-mono mb-0.5">z (pre-ReLU)</div>
                    <div className="flex flex-col gap-0.5">
                      {head.dense1.z.map((v, i) => (
                        <span key={i} className="font-mono text-[10px] text-gray-400">
                          {fmt(v)}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div>
                    <div className="text-[9px] text-gray-500 font-mono mb-0.5">relu</div>
                    <div className="flex flex-col gap-0.5">
                      {head.dense1.relu.map((v, i) => (
                        <span
                          key={i}
                          className="font-mono text-[10px]"
                          style={{ color: v > 0 ? "#4ade80" : "#4b5563" }}
                        >
                          {fmt(v)}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </Section>

            <Arrow />

            {/* ── Dense 2 ── */}
            <Section title="Dense 2" subtitle="(3 × 8)  →  logits" color="#60a5fa">
              <div className="flex flex-row gap-3 items-start">
                <MatrixDisplay matrix={head.dense2.W} label="W_d2" />
                <div className="flex flex-col gap-2 pt-3">
                  <div>
                    <div className="text-[9px] text-gray-500 font-mono mb-0.5">logits</div>
                    <div className="flex flex-col gap-0.5">
                      {head.dense2.logits.map((v, i) => (
                        <span
                          key={i}
                          className="font-mono text-[10px]"
                          style={{ color: CLASS_COLORS[i] }}
                        >
                          {fmt(v)}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </Section>

            <Arrow />

            {/* ── Softmax + Pastillas ── */}
            <Section title="Softmax" subtitle="3 clases" color="#a78bfa">
              <div className="flex flex-col gap-3 pt-1">
                {/* Pastillas */}
                <div className="flex flex-row gap-2">
                  {head.softmax.map((p, i) => {
                    const isWinner = i === winnerIdx;
                    return (
                      <div key={i} className="flex flex-col items-center gap-1">
                        <motion.div
                          className="rounded-lg px-4 py-2 text-xs font-bold text-center select-none"
                          style={{
                            minWidth: 76,
                            background: CLASS_BG[i],
                            border: `2px solid ${CLASS_COLORS[i]}`,
                            color: CLASS_COLORS[i],
                            opacity: 0.25 + p * 0.75,
                          }}
                          animate={
                            isWinner
                              ? {
                                  boxShadow: [
                                    `0 0 0px ${CLASS_COLORS[i]}00`,
                                    `0 0 14px ${CLASS_COLORS[i]}cc`,
                                    `0 0 0px ${CLASS_COLORS[i]}00`,
                                  ],
                                }
                              : {}
                          }
                          transition={
                            isWinner
                              ? { duration: 1.6, repeat: Infinity, ease: "easeInOut" }
                              : {}
                          }
                        >
                          {CLASS_LABELS[i]}
                        </motion.div>
                        <span
                          className="text-[11px] font-mono font-semibold"
                          style={{ color: CLASS_COLORS[i] }}
                        >
                          {fmtPct(p)}
                        </span>
                        {/* Barra de probabilidad */}
                        <div className="w-[76px] h-1.5 rounded-full bg-gray-800 overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{
                              width: `${p * 100}%`,
                              backgroundColor: CLASS_COLORS[i],
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Predicción */}
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-[10px] text-gray-500 font-mono">predicción:</span>
                  <span
                    className="text-sm font-bold font-mono capitalize"
                    style={{ color: CLASS_COLORS[winnerIdx] }}
                  >
                    {head.prediccion}
                  </span>
                </div>
              </div>
            </Section>

          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function Section({
  title,
  subtitle,
  color,
  children,
}: {
  title: string;
  subtitle: string;
  color: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1.5 p-3 rounded-lg bg-gray-900 border border-gray-800">
      <div className="flex flex-col">
        <span className="text-[11px] font-semibold font-mono" style={{ color }}>
          {title}
        </span>
        <span className="text-[9px] text-gray-500 font-mono">{subtitle}</span>
      </div>
      {children}
    </div>
  );
}

function Arrow() {
  return (
    <div className="self-center text-gray-600 text-base font-mono select-none pt-4">→</div>
  );
}
