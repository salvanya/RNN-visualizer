export function fmt(n: number): string {
  const s = n.toFixed(2);
  return n >= 0 ? `+${s}` : s;
}

export function fmtUnsigned(n: number): string {
  return n.toFixed(2);
}

// Compact display: drops leading zero. "0.12" → ".12", "-0.12" → "-.12".
export function fmtCompact(n: number): string {
  const s = n.toFixed(2);
  if (s.startsWith("0.")) return s.slice(1);
  if (s.startsWith("-0.")) return "-" + s.slice(2);
  return s;
}

export function fmtPct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}
