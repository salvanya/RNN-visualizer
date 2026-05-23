import { useEffect, useRef } from "react";
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

  return (
    <div className="flex items-center gap-1 px-4 py-2 bg-gray-900 border-b border-gray-700 overflow-x-auto">
      {steps.map((step) => {
        const active = step.index === timestep;
        const bgClass = (() => {
          if (active) {
            if (step.type === "enc") return "bg-blue-500 text-white ring-2 ring-blue-300";
            if (step.type === "dec") return "bg-orange-500 text-white ring-2 ring-orange-300";
            return "bg-gray-400 text-white ring-2 ring-gray-200";
          }
          if (step.type === "enc") return "bg-blue-900/70 text-blue-300 hover:bg-blue-800/70";
          if (step.type === "dec") return "bg-orange-900/70 text-orange-300 hover:bg-orange-800/70";
          return "bg-gray-700 text-gray-400 hover:bg-gray-600";
        })();

        return (
          <button
            key={step.index}
            onClick={() => setTimestep(step.index)}
            className={`shrink-0 px-2 py-1 rounded text-xs font-mono transition-all cursor-pointer select-none ${bgClass}`}
          >
            {step.label}
          </button>
        );
      })}
    </div>
  );
}
