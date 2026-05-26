import { useStore, type Arquitectura, type Atencion, type Modo } from "../state/store";

export default function Controls() {
  const {
    arquitectura, setArquitectura,
    modo, setModo,
    atencion, setAtencion,
    tooltipsFijos, limpiarTooltips,
  } = useStore();

  const isTranslation = modo === "translation";

  return (
    <div className="flex flex-wrap items-center gap-4 px-6 py-3 bg-gray-900 border-b border-gray-700 text-sm text-gray-200">

      {/* Arquitectura */}
      <div className="flex items-center gap-1">
        <span className="text-gray-400 mr-1">Arquitectura:</span>
        {(["GRU", "LSTM"] as Arquitectura[]).map((a) => (
          <button
            key={a}
            onClick={() => setArquitectura(a)}
            className={`px-3 py-1 rounded text-xs font-semibold transition-colors ${
              arquitectura === a
                ? "bg-blue-600 text-white"
                : "bg-gray-700 text-gray-300 hover:bg-gray-600"
            }`}
          >
            {a}
          </button>
        ))}
      </div>

      {/* Modo */}
      <div className="flex items-center gap-1">
        <span className="text-gray-400 mr-1">Modo:</span>
        {([
          ["sentiment", "Sentiment"],
          ["translation", "Traducción"],
        ] as [Modo, string][]).map(([m, label]) => (
          <button
            key={m}
            onClick={() => setModo(m)}
            className={`px-3 py-1 rounded text-xs font-semibold transition-colors ${
              modo === m
                ? "bg-indigo-600 text-white"
                : "bg-gray-700 text-gray-300 hover:bg-gray-600"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Atención (solo translation) — radio tri-state */}
      {isTranslation && (
        <div className="flex items-center gap-1">
          <span className="text-gray-400 mr-1">Atención:</span>
          {([
            ["none", "Sin atención"],
            ["bahdanau", "Bahdanau"],
            ["luong", "Luong"],
          ] as [Atencion, string][]).map(([a, label]) => (
            <button
              key={a}
              onClick={() => setAtencion(a)}
              className={`px-3 py-1 rounded text-xs font-semibold transition-colors ${
                atencion === a
                  ? "bg-orange-500 text-white"
                  : "bg-gray-700 text-gray-300 hover:bg-gray-600"
              }`}
              title={
                a === "none"
                  ? "Decoder ve solo el contexto comprimido del encoder"
                  : a === "luong"
                  ? "Atención multiplicativa, output-side"
                  : "Atención aditiva, input-side (modifica la dinámica del decoder)"
              }
            >
              {label}
            </button>
          ))}
        </div>
      )}

      {/* Limpiar tooltips */}
      {tooltipsFijos.length > 0 && (
        <button
          onClick={limpiarTooltips}
          className="ml-auto px-3 py-1 rounded text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors"
        >
          Limpiar tooltips ({tooltipsFijos.length})
        </button>
      )}
    </div>
  );
}
