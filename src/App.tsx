import Controls from "./components/Controls";
import TimestepBar from "./components/TimestepBar";
import InputSentence from "./components/InputSentence";
import OutputSentence from "./components/OutputSentence";
import Encoder from "./components/Encoder";
import Decoder from "./components/Decoder";
import SentimentHead from "./components/SentimentHead";
import CellInternalModal from "./components/CellInternalModal";
import { SectionHeader } from "./components/Section";
import { useStore } from "./state/store";

export default function App() {
  const { modo } = useStore();
  const isTranslation = modo === "translation";

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      {/* Barra superior */}
      <header className="shrink-0">
        <Controls />

        {/* Frase de salida (solo translation) — arriba de la barra de timesteps */}
        {isTranslation && (
          <div className="px-6 py-2 bg-gray-900 border-b border-gray-800">
            <OutputSentence />
          </div>
        )}

        <TimestepBar />
      </header>

      {/* Cuerpo principal */}
      <main className="flex-1 flex flex-col gap-7 p-6 overflow-auto">
        {/* Frase de entrada */}
        <InputSentence />

        {/* Encoder */}
        <section>
          <SectionHeader title="Encoder" color="#60a5fa" subtitle="2 capas RNN apiladas" />
          <Encoder />
        </section>

        {/* Head o Decoder */}
        {!isTranslation ? (
          <section>
            <SectionHeader title="Sentiment Head" color="#4ade80" subtitle="densas + softmax (3 clases)" />
            <SentimentHead />
          </section>
        ) : (
          <section>
            <SectionHeader title="Decoder" color="#fb923c" subtitle="autoregresivo, 2 capas" />
            <Decoder />
          </section>
        )}
      </main>

      {/* Modal global de celda interna */}
      <CellInternalModal />
    </div>
  );
}
