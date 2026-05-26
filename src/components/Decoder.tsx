import { useStore } from '../state/store';
import { appData, getDecoderTimesteps, getTranslationScenario } from '../data/index';
import { decoderTokenColor } from '../utils/colors';
import LayerStack, { type LayerStackTimestep } from './LayerStack';
import ContextProjection from './ContextProjection';
import SoftmaxHead from './SoftmaxHead';
import AttentionLayer from './AttentionLayer';
import { SectionHeader, WaitingMessage } from './Section';

export default function Decoder() {
  const { timestep, arquitectura, atencion } = useStore();
  const scenario = getTranslationScenario(arquitectura, atencion);
  const decTimesteps = getDecoderTimesteps(arquitectura, atencion);
  const { config } = appData;
  const isLstm = arquitectura === 'LSTM';

  const visibleCount = Math.min(Math.max(timestep - 7, 0), 6);
  const activeT = timestep >= 8 && timestep <= 13 ? timestep - 7 : null;

  const weights = scenario.decoder.weights;
  const isBahdanau = atencion === "bahdanau";
  // En Bahdanau, decoder L1 recibe [embedding ; c_t] (dim D+L=8) en vez de solo embedding (dim D=4).
  const inputDimL1 = isBahdanau ? config.d + config.l : config.d;

  const stackTimesteps: LayerStackTimestep[] = decTimesteps.map((ts, i) => {
    const predicted = ts.softmax.argmax;
    // Bahdanau: el input "real" a decoder L1 es la concatenación [emb; c_t]
    const inputVec = (ts as { input_concat?: number[] }).input_concat ?? ts.input_embedding;
    return {
      t: i + 1,
      tokenLabel: predicted,
      inputTokenLabel: ts.input_token,
      tokenColor: decoderTokenColor(predicted),
      inputVector: inputVec,
      layer1: ts.layer1,
      layer2: ts.layer2,
    };
  });

  const attentionBlock = visibleCount > 0 && atencion !== "none" && (
    <div>
      <SectionHeader
        title="Mecanismo de atención"
        color="#fbbf24"
        subtitle={
          isBahdanau
            ? "Bahdanau aditiva — entra al input del decoder"
            : "Luong general — decora la salida del decoder"
        }
        size="sm"
      />
      <AttentionLayer />
    </div>
  );

  return (
    <div className="flex flex-col gap-5">
      <ContextProjection />

      {/* Bahdanau: atención ARRIBA del decoder (input-side) */}
      {isBahdanau && attentionBlock}

      <div className="overflow-x-auto overflow-y-visible py-4 px-3 pt-7">
        <LayerStack
          side="dec"
          accentColor="#fb923c"
          layer1Units={config.p}
          layer2Units={config.q}
          inputDimL1={inputDimL1}
          embeddingDim={isBahdanau ? config.d : undefined}
          visibleCount={visibleCount}
          activeT={activeT}
          timesteps={stackTimesteps}
          weightsL1={weights.layer1}
          weightsL2={weights.layer2}
          isLstm={isLstm}
        />
        {timestep === 7 && visibleCount === 0 && (
          <WaitingMessage className="mt-3 ml-3">
            Presiona → para comenzar el decoder
          </WaitingMessage>
        )}
      </div>

      {/* Luong: atención DEBAJO del decoder (output-side) */}
      {!isBahdanau && attentionBlock}

      {visibleCount > 0 && (
        <div>
          <SectionHeader title="Proyección sobre vocabulario" color="#fb923c" subtitle="softmax sobre 12 tokens" size="sm" />
          <SoftmaxHead />
        </div>
      )}
    </div>
  );
}
