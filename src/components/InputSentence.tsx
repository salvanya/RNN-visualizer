import { useStore } from '../state/store';
import { appData } from '../data/index';
import TokenPill from './TokenPill';

export default function InputSentence() {
  const { timestep } = useStore();
  const tokens = appData.config.encoderTokens;
  const embeddings = appData.config.embeddings;

  // timestep 1..7 → active token index 0..6
  const activeIdx = timestep >= 1 && timestep <= 7 ? timestep - 1 : null;

  return (
    <div className="flex flex-col gap-2">
      <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">
        Entrada — español
      </div>
      <div className="flex items-start gap-3 flex-wrap">
        {tokens.map((token, i) => (
          <TokenPill
            key={token}
            token={token}
            embedding={embeddings[token] ?? []}
            isActive={activeIdx === i}
            isProcessed={activeIdx !== null ? i < activeIdx : timestep > 7}
          />
        ))}
      </div>
    </div>
  );
}
