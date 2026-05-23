export function fmt(n: number): string {
  const s = n.toFixed(2);
  return n >= 0 ? `+${s}` : s;
}

export function fmtUnsigned(n: number): string {
  return n.toFixed(2);
}

export function fmtPct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}
