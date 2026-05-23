import type { Velocidad } from "../state/store";

const BASE_MS = {
  embeddingAparece:       400,
  vectorViajando:         600,
  gateIluminandose:       300,
  desdoblamientoTimestep: 500,
  aperturaModal:          200,
  cambioTooltip:          150,
} as const;

export type AnimKey = keyof typeof BASE_MS;

export function duration(key: AnimKey, velocidad: Velocidad): number {
  return BASE_MS[key] / velocidad;
}

// Duración en segundos (para Framer Motion)
export function durationSec(key: AnimKey, velocidad: Velocidad): number {
  return duration(key, velocidad) / 1000;
}

// Intervalo de auto-avance en ms
export function autoAdvanceInterval(velocidad: Velocidad): number {
  return BASE_MS.desdoblamientoTimestep / velocidad + 200;
}
