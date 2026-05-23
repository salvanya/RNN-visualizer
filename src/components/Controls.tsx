import { Play, Pause, Square } from "lucide-react";
import { useStore, type Arquitectura, type Modo, type Velocidad } from "../state/store";

export default function Controls() {
  const {
    arquitectura, setArquitectura,
    modo, setModo,
    atencion, setAtencion,
    playState, setPlayState,
    velocidad, setVelocidad,
    tooltipsFijos, limpiarTooltips,
  } = useStore();

  const isTranslation = modo === "translation";

  return (
    <div className="flex flex-wrap items-center gap-4 px-6 py-3 bg-gray-900 border-b border-gray-700 text-sm text-gray-200">

      {/* Arquitectura */}
      <div className="flex items-center gap-1">
        <span className="text-gray-400 mr-1">Arq:</span>
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

      {/* Atención (solo translation) */}
      {isTranslation && (
        <div className="flex items-center gap-2">
          <span className="text-gray-400">Atención:</span>
          <button
            onClick={() => setAtencion(!atencion)}
            className={`relative inline-flex h-5 w-10 items-center rounded-full transition-colors ${
              atencion ? "bg-orange-500" : "bg-gray-600"
            }`}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                atencion ? "translate-x-5" : "translate-x-1"
              }`}
            />
          </button>
        </div>
      )}

      <div className="h-5 border-l border-gray-600" />

      {/* Play / Pause / Stop */}
      <div className="flex items-center gap-1">
        {playState !== "playing" ? (
          <button
            onClick={() => setPlayState("playing")}
            className="p-1.5 rounded bg-gray-700 hover:bg-green-700 transition-colors"
            title="Reproducir (Espacio)"
          >
            <Play size={14} />
          </button>
        ) : (
          <button
            onClick={() => setPlayState("paused")}
            className="p-1.5 rounded bg-green-700 hover:bg-green-600 transition-colors"
            title="Pausar (Espacio)"
          >
            <Pause size={14} />
          </button>
        )}
        <button
          onClick={() => setPlayState("stopped")}
          className="p-1.5 rounded bg-gray-700 hover:bg-red-700 transition-colors"
          title="Detener"
        >
          <Square size={14} />
        </button>
      </div>

      {/* Velocidad */}
      <div className="flex items-center gap-1">
        <span className="text-gray-400 mr-1">Vel:</span>
        {([0.5, 1, 2] as Velocidad[]).map((v) => (
          <button
            key={v}
            onClick={() => setVelocidad(v)}
            className={`px-2 py-1 rounded text-xs font-mono transition-colors ${
              velocidad === v
                ? "bg-gray-500 text-white"
                : "bg-gray-700 text-gray-300 hover:bg-gray-600"
            }`}
          >
            {v}x
          </button>
        ))}
      </div>

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
