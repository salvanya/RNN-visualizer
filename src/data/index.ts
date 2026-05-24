import rawData from './data.json';
import type { RNNData, EncoderTimestep, DecoderTimestep, TranslationScenario } from './types';

export const appData = rawData as unknown as RNNData;

export function getEncoderTimesteps(
  arquitectura: 'GRU' | 'LSTM',
  modo: 'sentiment' | 'translation'
): EncoderTimestep[] {
  const key = modo === 'sentiment'
    ? `${arquitectura}_sentiment`
    : `${arquitectura}_translation_noattn`;
  const scenarios = appData.scenarios as unknown as Record<string, { encoder: { timesteps: EncoderTimestep[] } }>;
  return scenarios[key].encoder.timesteps;
}

export function getTranslationScenario(
  arquitectura: 'GRU' | 'LSTM',
  atencion: boolean
): TranslationScenario {
  const key = `${arquitectura}_translation_${atencion ? 'attn' : 'noattn'}`;
  return (appData.scenarios as unknown as Record<string, TranslationScenario>)[key];
}

export function getDecoderTimesteps(
  arquitectura: 'GRU' | 'LSTM',
  atencion: boolean
): DecoderTimestep[] {
  return getTranslationScenario(arquitectura, atencion).decoder.timesteps;
}
