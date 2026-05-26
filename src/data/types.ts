// ─── Config ────────────────────────────────────────────────────────────────

export interface Config {
  d: number;
  m: number;
  l: number;
  p: number;
  q: number;
  V: number;
  encoderTokens: string[];
  vocab: string[];
  embeddings: Record<string, number[]>;
}

// ─── Gates ────────────────────────────────────────────────────────────────

export interface GruGateDetail {
  W_ih_x: number[];
  W_hh_h: number[];
  sum: number[];
  sigmoid: number[];
}

export interface GruCandidateDetail {
  W_ih_x: number[];
  W_hh_rh: number[];
  sum: number[];
  tanh: number[];
}

export interface GruGates {
  reset: GruGateDetail;
  update: GruGateDetail;
  candidate: GruCandidateDetail;
}

export interface LstmGateDetail {
  W_ih_x: number[];
  W_hh_h: number[];
  sum: number[];
  sigmoid?: number[];
  tanh?: number[];
}

export interface LstmGates {
  forget: LstmGateDetail;
  input: LstmGateDetail;
  candidate: LstmGateDetail;
  output: LstmGateDetail;
  c_t: number[];
}

export type Gates = GruGates | LstmGates;

// ─── Encoder layers ────────────────────────────────────────────────────────

export interface EncoderLayerStateGru {
  h_prev: number[];
  x_t: number[];
  gates: GruGates;
  h_t: number[];
}

export interface EncoderLayerStateLstm {
  h_prev: number[];
  c_prev: number[];
  x_t: number[];
  gates: LstmGates;
  h_t: number[];
  c_t: number[];
}

export type EncoderLayerState = EncoderLayerStateGru | EncoderLayerStateLstm;

export function isLstmLayer(s: EncoderLayerState): s is EncoderLayerStateLstm {
  return "c_prev" in s;
}

export interface EncoderTimestepGru {
  t: number;
  input: string;
  layer1: EncoderLayerStateGru;
  layer2: EncoderLayerStateGru;
}

export interface EncoderTimestepLstm {
  t: number;
  input: string;
  layer1: EncoderLayerStateLstm;
  layer2: EncoderLayerStateLstm;
}

export type EncoderTimestep = EncoderTimestepGru | EncoderTimestepLstm;

// ─── GRU weights ──────────────────────────────────────────────────────────

export interface GruLayerWeights {
  W_ih_r: number[][];
  W_hh_r: number[][];
  b_r: number[];
  W_ih_z: number[][];
  W_hh_z: number[][];
  b_z: number[];
  W_ih_n: number[][];
  W_hh_n: number[][];
  b_n: number[];
}

export interface LstmLayerWeights {
  W_ih_f: number[][];
  W_hh_f: number[][];
  b_f: number[];
  W_ih_i: number[][];
  W_hh_i: number[][];
  b_i: number[];
  W_ih_g: number[][];
  W_hh_g: number[][];
  b_g: number[];
  W_ih_o: number[][];
  W_hh_o: number[][];
  b_o: number[];
}

// ─── Encoder scenario ──────────────────────────────────────────────────────

export interface EncoderScenario {
  weights: {
    layer1: GruLayerWeights | LstmLayerWeights;
    layer2: GruLayerWeights | LstmLayerWeights;
  };
  timesteps: EncoderTimestep[];
  h2_all?: number[][];
}

// ─── Sentiment head ────────────────────────────────────────────────────────

export interface SentimentHead {
  input: number[];
  dense1: {
    W: number[][];
    b: number[];
    z: number[];
    relu: number[];
  };
  dense2: {
    W: number[][];
    b: number[];
    logits: number[];
  };
  softmax: number[];
  prediccion: "negativo" | "neutro" | "positivo";
  weights: {
    W_d1: number[][];
    b_d1: number[];
    W_d2: number[][];
    b_d2: number[];
  };
}

// ─── Decoder ──────────────────────────────────────────────────────────────

export interface SoftmaxOutput {
  logits: number[];
  probas: number[];
  argmax: string;
}

export interface AttentionOutputLuong {
  scores: number[];
  alphas: number[];
  contexto: number[];
  h_tilde: number[];
}

export interface BahdanauAttentionDetail {
  W_a_h_dec: number[];
  U_a_h_enc: number[][];
  pre_tanh: number[][];
  tanh: number[][];
}

export interface AttentionOutputBahdanau {
  tipo: "bahdanau";
  h_dec_prev: number[];
  scores: number[];
  alphas: number[];
  contexto: number[];
  detail: BahdanauAttentionDetail;
}

export type AttentionOutput = AttentionOutputLuong | AttentionOutputBahdanau;

export function isBahdanauAttention(
  a: AttentionOutput
): a is AttentionOutputBahdanau {
  return "tipo" in a && a.tipo === "bahdanau";
}

export interface DecoderTimestepGruNoAttn {
  t: number;
  input_token: string;
  input_embedding: number[];
  layer1: EncoderLayerStateGru;
  layer2: EncoderLayerStateGru;
  softmax: SoftmaxOutput;
}

export interface DecoderTimestepGruAttn {
  t: number;
  input_token: string;
  input_embedding: number[];
  // input_concat solo existe en Bahdanau: [embedding ; c_t], dim D+L
  input_concat?: number[];
  layer1: EncoderLayerStateGru;
  layer2: EncoderLayerStateGru;
  attention: AttentionOutput;
  softmax: SoftmaxOutput;
}

export interface DecoderTimestepLstmNoAttn {
  t: number;
  input_token: string;
  input_embedding: number[];
  layer1: EncoderLayerStateLstm;
  layer2: EncoderLayerStateLstm;
  softmax: SoftmaxOutput;
}

export interface DecoderTimestepLstmAttn {
  t: number;
  input_token: string;
  input_embedding: number[];
  input_concat?: number[];
  layer1: EncoderLayerStateLstm;
  layer2: EncoderLayerStateLstm;
  attention: AttentionOutput;
  softmax: SoftmaxOutput;
}

export type DecoderTimestep =
  | DecoderTimestepGruNoAttn
  | DecoderTimestepGruAttn
  | DecoderTimestepLstmNoAttn
  | DecoderTimestepLstmAttn;

export function hasAttention(
  ts: DecoderTimestep
): ts is DecoderTimestepGruAttn | DecoderTimestepLstmAttn {
  return "attention" in ts;
}

export interface ContextInfo {
  W_c: number[][];
  h_T_layer2: number[];
  h_0_decoder_layer1: number[];
}

export interface DecoderScenario {
  weights: {
    layer1: GruLayerWeights | LstmLayerWeights;
    layer2: GruLayerWeights | LstmLayerWeights;
  };
  W_out: number[][];
  // Luong: W_a (Q×L), W_combine (Q × (Q+L))
  W_a?: number[][];
  W_combine?: number[][];
  // Bahdanau: W_a (A×Q), U_a (A×L), v_a (A)
  U_a?: number[][];
  v_a?: number[];
  timesteps: DecoderTimestep[];
}

// ─── Scenarios ────────────────────────────────────────────────────────────

export interface SentimentScenario {
  encoder: EncoderScenario;
  head: SentimentHead;
}

export interface TranslationScenario {
  encoder: EncoderScenario;
  context: ContextInfo;
  decoder: DecoderScenario;
}

export interface Scenarios {
  GRU_sentiment: SentimentScenario;
  LSTM_sentiment: SentimentScenario;
  GRU_translation_noattn: TranslationScenario;
  GRU_translation_attn: TranslationScenario;
  GRU_translation_attn_bahdanau: TranslationScenario;
  LSTM_translation_noattn: TranslationScenario;
  LSTM_translation_attn: TranslationScenario;
  LSTM_translation_attn_bahdanau: TranslationScenario;
}

export interface RNNData {
  config: Config;
  scenarios: Scenarios;
}
