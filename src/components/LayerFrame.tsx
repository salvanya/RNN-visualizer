interface Props {
  title: string;
  subtitle?: string;
  color: string;
  top: number;
  height: number;
  width: number;
}

export default function LayerFrame({ title, subtitle, color, top, height, width }: Props) {
  return (
    <div
      className="absolute rounded-2xl pointer-events-none"
      style={{
        top,
        left: 0,
        width,
        height,
        background: `linear-gradient(180deg, ${color}0d 0%, ${color}06 100%)`,
        border: `1px solid ${color}26`,
        boxShadow: `inset 0 0 0 1px ${color}11`,
        zIndex: 12,
      }}
    >
      <div
        className="absolute -top-2.5 left-4 flex items-baseline gap-2 px-2.5 py-0.5 rounded-md"
        style={{ background: '#0a0a0a', border: `1px solid ${color}40`, zIndex: 15 }}
      >
        <span
          className="font-semibold uppercase tracking-[0.12em] text-[10px]"
          style={{ color }}
        >
          {title}
        </span>
        {subtitle && (
          <span className="text-[9px] text-gray-500 font-mono normal-case tracking-normal">
            {subtitle}
          </span>
        )}
      </div>
    </div>
  );
}
