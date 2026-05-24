import { ArrowRight } from 'lucide-react';
import type { EncoderTimestep } from '../data/types';
import { isLstmLayer } from '../data/types';
import { appData } from '../data/index';
import { useStore } from '../state/store';
import Cell from './Cell';

interface Props {
  layerNum: 1 | 2;
  encTimesteps: EncoderTimestep[];
  visibleCount: number;
  activeT: number | null;
}

function GruArrow() {
  return (
    <div className="px-1 text-gray-600 flex items-center">
      <ArrowRight size={12} />
    </div>
  );
}

function LstmArrow() {
  return (
    <div className="px-1.5 flex flex-col gap-0.5 items-center justify-center">
      <div className="flex items-center gap-0.5">
        <span className="font-mono text-[7px] text-blue-400 leading-none">h</span>
        <ArrowRight size={9} className="text-blue-400" />
      </div>
      <div className="flex items-center gap-0.5">
        <span className="font-mono text-[7px] text-yellow-400 leading-none">C</span>
        <ArrowRight size={9} className="text-yellow-400" />
      </div>
    </div>
  );
}

export default function Layer({ layerNum, encTimesteps, visibleCount, activeT }: Props) {
  const { arquitectura } = useStore();
  const isLstm = arquitectura === 'LSTM';
  const tokens = appData.config.encoderTokens;
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
          <span className="text-gray-700 text-xs font-mono italic">
            {isLstm ? 'h₀ = C₀ = [0…0]' : 'h₀ = [0…0]'}
          </span>
        ) : (
          Array.from({ length: visibleCount }, (_, i) => {
            const t = i + 1;
            const tsData = encTimesteps[i];
            const layerState = layerNum === 1 ? tsData.layer1 : tsData.layer2;
            const token = tokens[i];

            return (
              <div key={t} className="flex items-center">
                {i > 0 && (isLstm ? <LstmArrow /> : <GruArrow />)}
                <Cell
                  layer={layerNum}
                  t={t}
                  layerState={layerState}
                  token={token}
                  isActive={activeT === t}
                  tooltipSide={tooltipSide as 'above' | 'below'}
                  showCellState={isLstmLayer(layerState)}
                />
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
