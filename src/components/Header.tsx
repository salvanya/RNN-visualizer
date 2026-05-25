export default function Header() {
  const base = import.meta.env.BASE_URL;
  return (
    <div className="flex items-center justify-between px-6 py-3 bg-gray-950 border-b border-gray-800/80">
      <div className="flex items-center gap-3">
        <img
          src={`${base}logo.png`}
          alt="Logo Visualizador de RNN"
          className="h-10 w-10 rounded-md object-cover ring-1 ring-cyan-400/20"
        />
        <h1
          className="text-[22px] text-gray-50 font-semibold leading-none"
          style={{
            fontFamily: '"Space Grotesk", Inter, sans-serif',
            letterSpacing: '-0.02em',
          }}
        >
          Visualizador de <span className="text-cyan-300">RNN</span>
        </h1>
      </div>

      <div
        className="text-[11px] text-gray-500 tracking-wide"
        style={{ fontFamily: '"JetBrains Mono", monospace' }}
      >
        Sitio creado por{' '}
        <span className="text-gray-300 font-medium">Leandro Salvañá</span>
      </div>
    </div>
  );
}
