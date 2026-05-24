import { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { useStore } from "../state/store";
import { autoAdvanceInterval } from "../utils/timing";

// Timestep encoding:
//   0        = t_0 (gris)
//   1..7     = t_1^(e)..t_7^(e) (azul)
//   8..13    = t_1^(d)..t_6^(d) (naranja) — solo en translation

interface StepDef {
  index: number;
  label: string;
  type: "init" | "enc" | "dec";
}

function buildSteps(isTranslation: boolean): StepDef[] {
  const steps: StepDef[] = [{ index: 0, label: "t₀", type: "init" }];
  for (let i = 1; i <= 7; i++) {
    steps.push({ index: i, label: `t${i}ᵉ`, type: "enc" });
  }
  if (isTranslation) {
    for (let i = 1; i <= 6; i++) {
      steps.push({ index: 7 + i, label: `t${i}ᵈ`, type: "dec" });
    }
  }
  return steps;
}

export default function TimestepBar() {
  const { modo, timestep, setTimestep, playState, avanzarTimestep, velocidad } = useStore();
  const isTranslation = modo === "translation";
  const steps = buildSteps(isTranslation);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isDragging = useRef(false);

  // Auto-advance when playing
  useEffect(() => {
    if (playState === "playing") {
      const ms = autoAdvanceInterval(velocidad);
      intervalRef.current = setInterval(() => {
        avanzarTimestep();
      }, ms);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [playState, velocidad, avanzarTimestep]);

  // Keyboard shortcuts
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "ArrowRight") {
        e.preventDefault();
        avanzarTimestep();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        useStore.getState().retrocederTimestep();
      } else if (e.key === " ") {
        e.preventDefault();
        const { playState: ps, setPlayState } = useStore.getState();
        if (ps === "playing") setPlayState("paused");
        else setPlayState("playing");
      } else if (e.key === "Escape") {
        e.preventDefault();
        useStore.getState().limpiarTooltips();
        useStore.getState().cerrarModalCelda();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [avanzarTimestep]);

  // Drag scrubbing: find which step button the pointer is over
  function getStepFromPointer(clientX: number, clientY: number): number | null {
    const el = document.elementFromPoint(clientX, clientY);
    if (!el) return null;
    const btn = (el as HTMLElement).closest("[data-step-index]") as HTMLElement | null;
    if (!btn) return null;
    const idx = parseInt(btn.dataset.stepIndex!, 10);
    return isNaN(idx) ? null : idx;
  }

  function onPointerDown(e: React.PointerEvent<HTMLDivElement>) {
    isDragging.current = true;
    (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);
    const idx = getStepFromPointer(e.clientX, e.clientY);
    if (idx !== null) setTimestep(idx);
  }

  function onPointerMove(e: React.PointerEvent<HTMLDivElement>) {
    if (!isDragging.current) return;
    const idx = getStepFromPointer(e.clientX, e.clientY);
    if (idx !== null) setTimestep(idx);
  }

  function onPointerUp(e: React.PointerEvent<HTMLDivElement>) {
    isDragging.current = false;
    (e.currentTarget as HTMLDivElement).releasePointerCapture(e.pointerId);
  }

  const isPlaying = playState === "playing";

  return (
    <div
      className="flex items-center gap-1 px-4 py-2 bg-gray-900 border-b border-gray-700 overflow-x-auto select-none cursor-pointer"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerLeave={(e) => {
        if (isDragging.current) {
          isDragging.current = false;
          (e.currentTarget as HTMLDivElement).releasePointerCapture(e.pointerId);
        }
      }}
    >
      {steps.map((step) => {
        const active = step.index === timestep;

        const baseClass = (() => {
          if (active) {
            if (step.type === "enc") return "bg-blue-500 text-white";
            if (step.type === "dec") return "bg-orange-500 text-white";
            return "bg-gray-400 text-white";
          }
          if (step.type === "enc") return "bg-blue-900/70 text-blue-300 hover:bg-blue-800/70";
          if (step.type === "dec") return "bg-orange-900/70 text-orange-300 hover:bg-orange-800/70";
          return "bg-gray-700 text-gray-400 hover:bg-gray-600";
        })();

        const ringClass = (() => {
          if (!active) return "";
          if (isPlaying) {
            if (step.type === "enc") return "ring-2 ring-blue-300";
            if (step.type === "dec") return "ring-2 ring-orange-300";
            return "ring-2 ring-gray-200";
          }
          if (step.type === "enc") return "ring-2 ring-blue-300/60";
          if (step.type === "dec") return "ring-2 ring-orange-300/60";
          return "ring-2 ring-gray-200/60";
        })();

        return (
          <div
            key={step.index}
            data-step-index={step.index}
            className={`relative shrink-0 px-2 py-1 rounded text-xs font-mono transition-colors ${baseClass} ${ringClass}`}
          >
            {step.label}
            {/* Pulsing dot when playing and active */}
            {active && isPlaying && (
              <motion.span
                className="absolute -top-1 -right-1 w-2 h-2 rounded-full"
                style={{
                  backgroundColor:
                    step.type === "enc" ? "#93c5fd" :
                    step.type === "dec" ? "#fdba74" : "#d1d5db",
                }}
                animate={{ opacity: [1, 0.2, 1], scale: [1, 1.4, 1] }}
                transition={{ duration: 0.9, repeat: Infinity, ease: "easeInOut" }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
