import rawData from './data.json';
import type { RNNData, EncoderTimestep, DecoderTimestep, TranslationScenario } from './types';
import type { Atencion } from '../state/store';

export const appData = rawData as unknown as RNNData;

function attnSuffix(atencion: Atencion): string {
  if (atencion === "none") return "noattn";
  if (atencion === "bahdanau") return "attn_bahdanau";
  return "attn"; // luong
}

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
  atencion: Atencion
): TranslationScenario {
  const key = `${arquitectura}_translation_${attnSuffix(atencion)}`;
  return (appData.scenarios as unknown as Record<string, TranslationScenario>)[key];
}

export function getDecoderTimesteps(
  arquitectura: 'GRU' | 'LSTM',
  atencion: Atencion
): DecoderTimestep[] {
  return getTranslationScenario(arquitectura, atencion).decoder.timesteps;
}
