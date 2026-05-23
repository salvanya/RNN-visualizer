// Paleta de tokens de entrada (encoder) — 7 colores distinguibles
export const ENCODER_TOKEN_COLORS: Record<string, string> = {
  Me:       "#2dd4bf", // turquesa
  encantó:  "#6366f1", // índigo
  la:       "#f97316", // naranja
  pizza:    "#a855f7", // púrpura
  de:       "#84cc16", // verde lima
  este:     "#ec4899", // rosa
  lugar:    "#f59e0b", // ámbar
};

// Paleta para tokens de salida (decoder)
export const DECODER_TOKEN_COLORS: Record<string, string> = {
  "<START>": "#94a3b8",
  "<END>":   "#64748b",
  I:         "#fbbf24",
  loved:     "#fb923c",
  this:      "#f87171",
  "place's": "#e879f9",
  pizza:     "#c084fc",
};

export function encoderTokenColor(token: string, opacity = 1): string {
  const hex = ENCODER_TOKEN_COLORS[token] ?? "#94a3b8";
  if (opacity === 1) return hex;
  return hexToRgba(hex, opacity);
}

export function decoderTokenColor(token: string, opacity = 1): string {
  const hex = DECODER_TOKEN_COLORS[token] ?? "#94a3b8";
  if (opacity === 1) return hex;
  return hexToRgba(hex, opacity);
}

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// Colores de heatmap para matrices (rojo negativo, blanco cero, verde positivo)
export function heatmapColor(value: number, min = -1, max = 1): string {
  const t = (value - min) / (max - min); // 0..1
  if (t < 0.5) {
    // rojo → blanco
    const f = t * 2;
    const r = 220;
    const g = Math.round(f * 220);
    const b = Math.round(f * 220);
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    // blanco → verde
    const f = (t - 0.5) * 2;
    const r = Math.round((1 - f) * 220);
    const g = 220;
    const b = Math.round((1 - f) * 220);
    return `rgb(${r}, ${g}, ${b})`;
  }
}
