import Header from "./components/Header";
import Controls from "./components/Controls";
import DimensionsLegend from "./components/DimensionsLegend";
import TimestepBar from "./components/TimestepBar";
import InputSentence from "./components/InputSentence";
import OutputSentence from "./components/OutputSentence";
import Encoder from "./components/Encoder";
import Decoder from "./components/Decoder";
import AttentionLayer from "./components/AttentionLayer";
import SentimentHead from "./components/SentimentHead";
import CellInternalModal from "./components/CellInternalModal";
import { SectionHeader } from "./components/Section";
import { useStore } from "./state/store";

export default function App() {
  const { modo, atencion, timestep } = useStore();
  const isTranslation = modo === "translation";
  const decoderStarted = timestep >= 8;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      {/* Barra superior */}
      <header className="shrink-0">
        <Header />
        <Controls />
        <DimensionsLegend />
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
          <>
            {/* Atención entre encoder y decoder (solo si está activada y el decoder ya arrancó) */}
            {atencion && decoderStarted && (
              <section>
                <SectionHeader title="Mecanismo de atención" color="#fbbf24" subtitle="Luong general — puente encoder ↔ decoder" />
                <AttentionLayer />
              </section>
            )}

            <section>
              <SectionHeader title="Decoder" color="#fb923c" subtitle="autoregresivo, 2 capas" />
              <Decoder />
            </section>

            {/* Colofón: frase generada como resultado final de la inferencia */}
            <section className="mt-2">
              <SectionHeader title="Salida (EN)" color="#fb923c" subtitle="traducción generada token a token" />
              <div className="px-4 py-3 rounded-lg bg-gray-900/70 border border-orange-900/40">
                <OutputSentence />
              </div>
            </section>
          </>
        )}
      </main>

      {/* Modal global de celda interna */}
      <CellInternalModal />
    </div>
  );
}
