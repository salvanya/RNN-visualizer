import Controls from "./components/Controls";
import TimestepBar from "./components/TimestepBar";
import InputSentence from "./components/InputSentence";
import OutputSentence from "./components/OutputSentence";
import Encoder from "./components/Encoder";
import Decoder from "./components/Decoder";
import SentimentHead from "./components/SentimentHead";
import CellInternalModal from "./components/CellInternalModal";
import { useStore } from "./state/store";

export default function App() {
  const { modo, timestep, arquitectura, atencion } = useStore();
  const isTranslation = modo === "translation";

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      {/* Barra superior */}
      <header className="shrink-0">
        <Controls />
        <TimestepBar />
      </header>

      {/* Estado actual (debug M1) */}
      <div className="shrink-0 px-6 py-2 bg-gray-900/50 border-b border-gray-800 text-xs text-gray-500 font-mono flex gap-6">
        <span>arq={arquitectura}</span>
        <span>modo={modo}</span>
        <span>atención={String(atencion)}</span>
        <span>timestep={timestep}</span>
      </div>

      {/* Frase de salida (solo translation) */}
      {isTranslation && (
        <div className="shrink-0 px-6 py-2 border-b border-gray-800">
          <OutputSentence />
        </div>
      )}

      {/* Cuerpo principal */}
      <main className="flex-1 flex flex-col gap-4 p-6 overflow-auto">
        {/* Frase de entrada */}
        <InputSentence />

        {/* Encoder */}
        <section>
          <h2 className="text-xs text-blue-400 font-semibold uppercase tracking-wider mb-2">
            Encoder
          </h2>
          <Encoder />
        </section>

        {/* Head o Decoder */}
        {!isTranslation ? (
          <section>
            <h2 className="text-xs text-green-400 font-semibold uppercase tracking-wider mb-2">
              Sentiment Head
            </h2>
            <SentimentHead />
          </section>
        ) : (
          <section>
            <h2 className="text-xs text-orange-400 font-semibold uppercase tracking-wider mb-2">
              Decoder
            </h2>
            <Decoder />
          </section>
        )}
      </main>
      {/* Modal global de celda interna */}
      <CellInternalModal />
    </div>
  );
}
