interface SectionHeaderProps {
  title: string;
  color: string;
  subtitle?: string;
  size?: 'lg' | 'sm';
}

export function SectionHeader({ title, color, subtitle, size = 'lg' }: SectionHeaderProps) {
  const isLarge = size === 'lg';
  return (
    <div className={`flex items-baseline gap-2 ${isLarge ? 'mb-3' : 'mb-2'}`}>
      <span
        className={`font-semibold uppercase tracking-[0.12em] ${
          isLarge ? 'text-[11px]' : 'text-[10px]'
        }`}
        style={{ color }}
      >
        {title}
      </span>
      {subtitle && (
        <span className="text-[10px] text-gray-500 font-mono normal-case tracking-normal">
          {subtitle}
        </span>
      )}
    </div>
  );
}

interface WaitingMessageProps {
  children: React.ReactNode;
  className?: string;
}

export function WaitingMessage({ children, className = '' }: WaitingMessageProps) {
  return (
    <div className={`text-gray-600 text-xs font-mono italic ${className}`}>
      {children}
    </div>
  );
}
