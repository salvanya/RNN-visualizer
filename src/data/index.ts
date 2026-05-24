import rawData from './data.json';
import type { RNNData, EncoderTimestep } from './types';

export const appData = rawData as unknown as RNNData;

export function getEncoderTimesteps(
  arquitectura: 'GRU' | 'LSTM',
  modo: 'sentiment' | 'translation'
): EncoderTimestep[] {
  const key = modo === 'sentiment'
    ? `${arquitectura}_sentiment`
    : `${arquitectura}_translation_noattn`;
  const scenarios = appData.scenarios as Record<string, { encoder: { timesteps: EncoderTimestep[] } }>;
  return scenarios[key].encoder.timesteps;
}
