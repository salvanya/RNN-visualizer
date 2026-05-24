import { ArrowRight } from 'lucide-react';
import type { EncoderTimestep } from '../data/types';
import { appData } from '../data/index';
import Cell from './Cell';

interface Props {
  layerNum: 1 | 2;
  encTimesteps: EncoderTimestep[];
  visibleCount: number;
  activeT: number | null;
}

export default function Layer({ layerNum, encTimesteps, visibleCount, activeT }: Props) {
  const tokens = appData.config.encoderTokens;
  // Layer 2 tooltip goes above (towards InputSentence), layer 1 goes below
  const tooltipSide = layerNum === 2 ? 'above' : 'below';

  return (
    <div className="flex items-center">
      {/* Layer label */}
      <div className="w-16 shrink-0 text-right pr-3">
        <span className="text-[11px] text-blue-400 font-mono">Capa {layerNum}</span>
      </div>

      {/* Cells */}
      <div className="flex items-center">
        {visibleCount === 0 ? (
          <span className="text-gray-700 text-xs font-mono italic">h₀ = [0…0]</span>
        ) : (
          Array.from({ length: visibleCount }, (_, i) => {
            const t = i + 1;
            const tsData = encTimesteps[i];
            const layerState = layerNum === 1 ? tsData.layer1 : tsData.layer2;
            const token = tokens[i];

            return (
              <div key={t} className="flex items-center">
                {i > 0 && (
                  <div className="px-1 text-gray-700 flex items-center">
                    <ArrowRight size={12} />
                  </div>
                )}
                <Cell
                  layer={layerNum}
                  t={t}
                  layerState={layerState}
                  token={token}
                  isActive={activeT === t}
                  tooltipSide={tooltipSide as 'above' | 'below'}
                />
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
