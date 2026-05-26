import { create } from "zustand";

export type Arquitectura = "GRU" | "LSTM";
export type Modo = "sentiment" | "translation";
export type Atencion = "none" | "luong" | "bahdanau";
export type PlayState = "stopped" | "playing" | "paused";
export type Velocidad = 0.5 | 1 | 2;

export type TooltipId = string;

export interface ModalCeldaId {
  layer: 1 | 2;
  t: number;
  lado: "enc" | "dec";
}

interface AppState {
  arquitectura: Arquitectura;
  modo: Modo;
  atencion: Atencion;
  // timestep: 0 = t_0, 1..7 = encoder, 8..13 = decoder
  timestep: number;
  playState: PlayState;
  velocidad: Velocidad;
  tooltipsFijos: TooltipId[];
  modalCeldaAbierta: ModalCeldaId | null;

  // Actions
  setArquitectura: (a: Arquitectura) => void;
  setModo: (m: Modo) => void;
  setAtencion: (v: Atencion) => void;
  setTimestep: (t: number) => void;
  avanzarTimestep: () => void;
  retrocederTimestep: () => void;
  setPlayState: (s: PlayState) => void;
  setVelocidad: (v: Velocidad) => void;
  fijarTooltip: (id: TooltipId) => void;
  liberarTooltip: (id: TooltipId) => void;
  limpiarTooltips: () => void;
  abrirModalCelda: (id: ModalCeldaId) => void;
  cerrarModalCelda: () => void;
}

function maxTimestep(modo: Modo): number {
  // 0 = t_0, 1..7 = encoder, 8..13 = decoder
  return modo === "sentiment" ? 7 : 13;
}

export const useStore = create<AppState>((set, get) => ({
  arquitectura: "GRU",
  modo: "sentiment",
  atencion: "none",
  timestep: 0,
  playState: "stopped",
  velocidad: 1,
  tooltipsFijos: [],
  modalCeldaAbierta: null,

  setArquitectura: (a) => set({ arquitectura: a }),

  setModo: (m) =>
    set((s) => ({
      modo: m,
      // Clamp timestep when switching to sentiment
      timestep: m === "sentiment" ? Math.min(s.timestep, 7) : s.timestep,
      playState: "stopped",
    })),

  setAtencion: (v) => set({ atencion: v }),

  setTimestep: (t) => {
    const { modo } = get();
    const max = maxTimestep(modo);
    set({ timestep: Math.max(0, Math.min(t, max)) });
  },

  avanzarTimestep: () => {
    const { timestep, modo, playState } = get();
    const max = maxTimestep(modo);
    if (timestep < max) {
      set({ timestep: timestep + 1 });
    } else if (playState === "playing") {
      // Reached end — stop (no cycling)
      set({ playState: "stopped" });
    }
  },

  retrocederTimestep: () => {
    const { timestep } = get();
    if (timestep > 0) set({ timestep: timestep - 1 });
  },

  setPlayState: (s) => {
    if (s === "stopped") {
      set({ playState: "stopped", timestep: 0 });
    } else {
      set({ playState: s });
    }
  },

  setVelocidad: (v) => set({ velocidad: v }),

  fijarTooltip: (id) =>
    set((s) => ({
      tooltipsFijos: s.tooltipsFijos.includes(id)
        ? s.tooltipsFijos
        : [...s.tooltipsFijos, id],
    })),

  liberarTooltip: (id) =>
    set((s) => ({
      tooltipsFijos: s.tooltipsFijos.filter((t) => t !== id),
    })),

  limpiarTooltips: () => set({ tooltipsFijos: [] }),

  abrirModalCelda: (id) => set({ modalCeldaAbierta: id }),

  cerrarModalCelda: () => set({ modalCeldaAbierta: null }),
}));
