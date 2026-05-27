import { useStore } from '../state/store';
import { appData, getEncoderTimesteps } from '../data/index';
import { encoderTokenColor } from '../utils/colors';
import LayerStack, { type LayerStackTimestep } from './LayerStack';
import EdgeLegend from './EdgeLegend';
import { WaitingMessage } from './Section';

export default function Encoder() {
  const { timestep, arquitectura, modo } = useStore();
  const isLstm = arquitectura === 'LSTM';

  const encTimesteps = getEncoderTimesteps(arquitectura, modo);
  const { config } = appData;
  const tokens = config.encoderTokens;
  const embeddings = config.embeddings;

  // Cells visible: 0 at t=0, grow to 7 by t=7, stay at 7 for decoder phase
  const visibleCount = Math.min(Math.max(timestep, 0), 7);
  const activeT = timestep >= 1 && timestep <= 7 ? timestep : null;

  // Locate weights from scenario
  const scenarioKey = modo === 'sentiment'
    ? `${arquitectura}_sentiment`
    : `${arquitectura}_translation_noattn`;
  const scenario = (appData.scenarios as unknown as Record<string, { encoder: { weights: { layer1: any; layer2: any } } }>)[scenarioKey];
  const weights = scenario.encoder.weights;

  const stackTimesteps: LayerStackTimestep[] = encTimesteps.map((ts, i) => {
    const token = tokens[i];
    return {
      t: i + 1,
      tokenLabel: token,
      tokenColor: encoderTokenColor(token),
      inputVector: embeddings[token] ?? ts.layer1.x_t,
      layer1: ts.layer1,
      layer2: ts.layer2,
    };
  });

  return (
    <div className="flex flex-col gap-4 py-4 px-3">
      <EdgeLegend showC={isLstm} />
      <div className="overflow-x-auto overflow-y-visible pt-3 app-scrollbar">
      <LayerStack
        side="enc"
        accentColor="#60a5fa"
        layer1Units={config.m}
        layer2Units={config.l}
        inputDimL1={config.d}
        visibleCount={visibleCount}
        activeT={activeT}
        timesteps={stackTimesteps}
        weightsL1={weights.layer1}
        weightsL2={weights.layer2}
        isLstm={isLstm}
      />
      </div>
      {visibleCount === 0 && (
        <WaitingMessage className="mt-3 ml-3">
          Presiona → o reproduce para comenzar
        </WaitingMessage>
      )}
    </div>
  );
}
