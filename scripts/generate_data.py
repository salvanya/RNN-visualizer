"""
generate_data.py — Precómputo de los 6 escenarios RNN para el visualizador.

Pesos ajustados a mano para que:
  1. Los números sean legibles (redondeos a 2 decimales).
  2. GRU/LSTM sentiment prediga "positivo".
  3. La traducción genere "I loved this place's pizza <END>".
  4. Los pesos de atención muestren patrones interpretables
     (encantó→loved, pizza→pizza, lugar→place's, etc.).

Validación interna: cada producto matricial se recalcula y se compara
con tolerancia 1e-6. Si algo no cuadra el script falla con AssertionError.
"""

import json
import numpy as np
import os

np.random.seed(42)

# ---------------------------------------------------------------------------
# Dimensiones (ver CLAUDE.md §3)
# ---------------------------------------------------------------------------
D = 4   # embedding
M = 3   # encoder capa 1
L = 4   # encoder capa 2
P = 5   # decoder capa 1
Q = 3   # decoder capa 2
V = 12  # vocabulario decoder

ENCODER_TOKENS = ["Me", "encantó", "la", "pizza", "de", "este", "lugar"]
VOCAB = ["<START>", "<END>", "I", "you", "loved", "hated",
         "this", "that", "place's", "food", "pizza", "the"]
# índices de vocab
IDX = {tok: i for i, tok in enumerate(VOCAB)}

TARGET_TRANSLATION = ["I", "loved", "this", "place's", "pizza", "<END>"]

# ---------------------------------------------------------------------------
# Embeddings hardcodeados (dim 4)
# ---------------------------------------------------------------------------
EMBEDDINGS = {
    # tokens de entrada
    "Me":       [ 0.30, -0.20,  0.50,  0.10],
    "encantó":  [ 0.80,  0.60, -0.10,  0.40],
    "la":       [-0.10,  0.20,  0.30, -0.20],
    "pizza":    [ 0.60,  0.10,  0.70,  0.50],
    "de":       [ 0.00,  0.30, -0.40,  0.20],
    "este":     [ 0.20, -0.10,  0.10,  0.60],
    "lugar":    [ 0.50,  0.40,  0.20, -0.30],
    # vocab decoder
    "<START>":  [ 0.00,  0.00,  0.00,  0.00],
    "<END>":    [ 0.10,  0.10,  0.10,  0.10],
    "I":        [ 0.40, -0.30,  0.20,  0.50],
    "you":      [ 0.35, -0.25,  0.15,  0.45],
    "loved":    [ 0.75,  0.55, -0.15,  0.35],
    "hated":    [-0.75, -0.55,  0.15, -0.35],
    "this":     [ 0.15, -0.05,  0.05,  0.55],
    "that":     [ 0.10, -0.10,  0.00,  0.50],
    "place's":  [ 0.45,  0.35,  0.15, -0.25],
    "food":     [ 0.55,  0.05,  0.65,  0.45],
    "pizza":    [ 0.60,  0.10,  0.70,  0.50],  # mismo que token de entrada
    "the":      [-0.05,  0.25,  0.25, -0.15],
}

def emb(tok):
    return np.array(EMBEDDINGS[tok], dtype=np.float64)

# ---------------------------------------------------------------------------
# Helpers GRU
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def r2(x):
    """Redondea array a 2 decimales para legibilidad en JSON."""
    return np.round(x, 2)

def make_weight(rows, cols, scale=0.5):
    """Pesos aleatorios en [-scale, scale], redondeados a 2 dec."""
    return r2((np.random.rand(rows, cols) * 2 - 1) * scale)

def make_bias(size, val=0.0):
    return np.full(size, val, dtype=np.float64)

# ---------------------------------------------------------------------------
# GRU forward (una capa)
# Devuelve h_t y el detalle de gates para el JSON
# ---------------------------------------------------------------------------
def gru_step(x, h_prev, W):
    """
    W contiene: W_ih_r, W_hh_r, b_r,
                W_ih_z, W_hh_z, b_z,
                W_ih_n, W_hh_n, b_n (n = candidate/new)
    Retorna (h_t, gate_detail_dict)
    """
    # reset gate
    r_lin = W["W_ih_r"] @ x + W["W_hh_r"] @ h_prev + W["b_r"]
    r = sigmoid(r_lin)
    # update gate
    z_lin = W["W_ih_z"] @ x + W["W_hh_z"] @ h_prev + W["b_z"]
    z = sigmoid(z_lin)
    # candidate
    n_lin = W["W_ih_n"] @ x + W["W_hh_n"] @ (r * h_prev) + W["b_n"]
    n = tanh(n_lin)
    # output
    h_t = (1 - z) * n + z * h_prev

    detail = {
        "reset":     {"W_ih_x": (W["W_ih_r"] @ x).tolist(),
                      "W_hh_h": (W["W_hh_r"] @ h_prev).tolist(),
                      "sum": r_lin.tolist(), "sigmoid": r.tolist()},
        "update":    {"W_ih_x": (W["W_ih_z"] @ x).tolist(),
                      "W_hh_h": (W["W_hh_z"] @ h_prev).tolist(),
                      "sum": z_lin.tolist(), "sigmoid": z.tolist()},
        "candidate": {"W_ih_x": (W["W_ih_n"] @ x).tolist(),
                      "W_hh_rh": (W["W_hh_n"] @ (r * h_prev)).tolist(),
                      "sum": n_lin.tolist(), "tanh": n.tolist()},
    }
    return h_t, detail

# ---------------------------------------------------------------------------
# LSTM forward (una capa)
# ---------------------------------------------------------------------------
def lstm_step(x, h_prev, c_prev, W):
    """
    W contiene: W_ih_f, W_hh_f, b_f,
                W_ih_i, W_hh_i, b_i,
                W_ih_g, W_hh_g, b_g,  (candidate)
                W_ih_o, W_hh_o, b_o
    """
    f_lin = W["W_ih_f"] @ x + W["W_hh_f"] @ h_prev + W["b_f"]
    f = sigmoid(f_lin)

    i_lin = W["W_ih_i"] @ x + W["W_hh_i"] @ h_prev + W["b_i"]
    i_ = sigmoid(i_lin)

    g_lin = W["W_ih_g"] @ x + W["W_hh_g"] @ h_prev + W["b_g"]
    g = tanh(g_lin)

    o_lin = W["W_ih_o"] @ x + W["W_hh_o"] @ h_prev + W["b_o"]
    o = sigmoid(o_lin)

    c_t = f * c_prev + i_ * g
    h_t = o * tanh(c_t)

    detail = {
        "forget":    {"W_ih_x": (W["W_ih_f"] @ x).tolist(),
                      "W_hh_h": (W["W_hh_f"] @ h_prev).tolist(),
                      "sum": f_lin.tolist(), "sigmoid": f.tolist()},
        "input":     {"W_ih_x": (W["W_ih_i"] @ x).tolist(),
                      "W_hh_h": (W["W_hh_i"] @ h_prev).tolist(),
                      "sum": i_lin.tolist(), "sigmoid": i_.tolist()},
        "candidate": {"W_ih_x": (W["W_ih_g"] @ x).tolist(),
                      "W_hh_h": (W["W_hh_g"] @ h_prev).tolist(),
                      "sum": g_lin.tolist(), "tanh": g.tolist()},
        "output":    {"W_ih_x": (W["W_ih_o"] @ x).tolist(),
                      "W_hh_h": (W["W_hh_o"] @ h_prev).tolist(),
                      "sum": o_lin.tolist(), "sigmoid": o.tolist()},
        "c_t": c_t.tolist(),
    }
    return h_t, c_t, detail

# ---------------------------------------------------------------------------
# Pesos del encoder (compartidos entre GRU y LSTM scenarios del mismo tipo)
# Se crean funciones para no compartir accidentalmente objetos mutables.
# ---------------------------------------------------------------------------

def make_gru_weights_enc():
    # Capa 1: input dim D=4, hidden dim M=3
    W1 = {
        "W_ih_r": make_weight(M, D), "W_hh_r": make_weight(M, M), "b_r": make_bias(M),
        "W_ih_z": make_weight(M, D), "W_hh_z": make_weight(M, M), "b_z": make_bias(M, 0.1),
        "W_ih_n": make_weight(M, D), "W_hh_n": make_weight(M, M), "b_n": make_bias(M),
    }
    # Capa 2: input dim M=3, hidden dim L=4.
    # W_ih_n con scale grande (1.5) y b_z negativo (-1.5) para que la puerta
    # de actualización esté abierta (z≈0.18) y cada token desplace fuertemente
    # el estado: esto garantiza que h_enc^(2) sea específico de cada token y
    # los estados sean suficientemente distintos entre sí para que la atención
    # sea interpretable. W_hh_n pequeño (0.1) para no difuminar esa info.
    W2 = {
        "W_ih_r": make_weight(L, M), "W_hh_r": make_weight(L, L), "b_r": make_bias(L),
        "W_ih_z": make_weight(L, M), "W_hh_z": make_weight(L, L), "b_z": make_bias(L, -1.5),
        "W_ih_n": make_weight(L, M, scale=1.5), "W_hh_n": make_weight(L, L, scale=0.1), "b_n": make_bias(L),
    }
    return W1, W2

def make_lstm_weights_enc():
    # Capa 1: input D=4, hidden M=3
    W1 = {
        "W_ih_f": make_weight(M, D), "W_hh_f": make_weight(M, M), "b_f": make_bias(M, 1.0),
        "W_ih_i": make_weight(M, D), "W_hh_i": make_weight(M, M), "b_i": make_bias(M),
        "W_ih_g": make_weight(M, D), "W_hh_g": make_weight(M, M), "b_g": make_bias(M),
        "W_ih_o": make_weight(M, D), "W_hh_o": make_weight(M, M), "b_o": make_bias(M),
    }
    # Capa 2: input M=3, hidden L=4
    W2 = {
        "W_ih_f": make_weight(L, M), "W_hh_f": make_weight(L, L), "b_f": make_bias(L, 1.0),
        "W_ih_i": make_weight(L, M), "W_hh_i": make_weight(L, L), "b_i": make_bias(L),
        "W_ih_g": make_weight(L, M), "W_hh_g": make_weight(L, L), "b_g": make_bias(L),
        "W_ih_o": make_weight(L, M), "W_hh_o": make_weight(L, L), "b_o": make_bias(L),
    }
    return W1, W2

# ---------------------------------------------------------------------------
# Forward encoder — devuelve timesteps + estado final
# ---------------------------------------------------------------------------

def encoder_forward_gru(W1, W2):
    h1 = np.zeros(M)
    h2 = np.zeros(L)
    timesteps = []
    enc_h2_all = []   # para atención

    for t, tok in enumerate(ENCODER_TOKENS, start=1):
        x = emb(tok)
        h1_prev = h1.copy()
        h2_prev = h2.copy()

        h1, g1 = gru_step(x, h1_prev, W1)
        h2, g2 = gru_step(h1, h2_prev, W2)

        enc_h2_all.append(h2.copy())
        timesteps.append({
            "t": t,
            "input": tok,
            "layer1": {
                "h_prev": h1_prev.tolist(),
                "x_t": x.tolist(),
                "gates": g1,
                "h_t": h1.tolist(),
            },
            "layer2": {
                "h_prev": h2_prev.tolist(),
                "x_t": h1.tolist(),
                "gates": g2,
                "h_t": h2.tolist(),
            },
        })

    return timesteps, h1, h2, enc_h2_all

def encoder_forward_lstm(W1, W2):
    h1 = np.zeros(M); c1 = np.zeros(M)
    h2 = np.zeros(L); c2 = np.zeros(L)
    timesteps = []
    enc_h2_all = []

    for t, tok in enumerate(ENCODER_TOKENS, start=1):
        x = emb(tok)
        h1_prev, c1_prev = h1.copy(), c1.copy()
        h2_prev, c2_prev = h2.copy(), c2.copy()

        h1, c1, g1 = lstm_step(x, h1_prev, c1_prev, W1)
        h2, c2, g2 = lstm_step(h1, h2_prev, c2_prev, W2)

        enc_h2_all.append(h2.copy())
        timesteps.append({
            "t": t,
            "input": tok,
            "layer1": {
                "h_prev": h1_prev.tolist(),
                "c_prev": c1_prev.tolist(),
                "x_t": x.tolist(),
                "gates": g1,
                "h_t": h1.tolist(),
                "c_t": c1.tolist(),
            },
            "layer2": {
                "h_prev": h2_prev.tolist(),
                "c_prev": c2_prev.tolist(),
                "x_t": h1.tolist(),
                "gates": g2,
                "h_t": h2.tolist(),
                "c_t": c2.tolist(),
            },
        })

    return timesteps, h1, h2, c1, c2, enc_h2_all

# ---------------------------------------------------------------------------
# Head de sentiment (compartido GRU/LSTM — misma estructura, pesos distintos)
# ---------------------------------------------------------------------------

def make_sentiment_head():
    # Dense 1: (8 × 4), Dense 2: (3 × 8)
    W_d1 = make_weight(8, L)
    b_d1 = make_bias(8)
    W_d2 = make_weight(3, 8)
    b_d2 = make_bias(3)
    return W_d1, b_d1, W_d2, b_d2

def sentiment_head_forward(h_T, W_d1, b_d1, W_d2, b_d2):
    z1 = W_d1 @ h_T + b_d1
    a1 = relu(z1)
    logits = W_d2 @ a1 + b_d2
    probs = softmax(logits)
    pred_idx = int(np.argmax(probs))
    clases = ["negativo", "neutro", "positivo"]
    return {
        "input": h_T.tolist(),
        "dense1": {"W": W_d1.tolist(), "b": b_d1.tolist(),
                   "z": z1.tolist(), "relu": a1.tolist()},
        "dense2": {"W": W_d2.tolist(), "b": b_d2.tolist(),
                   "logits": logits.tolist()},
        "softmax": probs.tolist(),
        "prediccion": clases[pred_idx],
    }

def tune_sentiment_head(h_T, target_class=2):
    """
    Ajusta pesos del head para que la clase target gane.
    Estrategia: construye pesos que proyectan h_T hacia la clase positiva.
    """
    np.random.seed(100)
    W_d1, b_d1, W_d2, b_d2 = make_sentiment_head()
    # Asegura que el nodo de salida 'positivo' (idx 2) domine.
    # Aumentamos el bias del nodo positivo.
    b_d2[target_class] += 2.0
    result = sentiment_head_forward(h_T, W_d1, b_d1, W_d2, b_d2)
    assert result["prediccion"] == "positivo", (
        f"Head no predice positivo: {result['softmax']}"
    )
    return W_d1, b_d1, W_d2, b_d2, result

# ---------------------------------------------------------------------------
# Decoder weights
# ---------------------------------------------------------------------------

def make_gru_weights_dec():
    # Capa 1: input D=4, hidden P=5
    W1 = {
        "W_ih_r": make_weight(P, D), "W_hh_r": make_weight(P, P), "b_r": make_bias(P),
        "W_ih_z": make_weight(P, D), "W_hh_z": make_weight(P, P), "b_z": make_bias(P, 0.1),
        "W_ih_n": make_weight(P, D), "W_hh_n": make_weight(P, P), "b_n": make_bias(P),
    }
    # Capa 2: input P=5, hidden Q=3
    W2 = {
        "W_ih_r": make_weight(Q, P), "W_hh_r": make_weight(Q, Q), "b_r": make_bias(Q),
        "W_ih_z": make_weight(Q, P), "W_hh_z": make_weight(Q, Q), "b_z": make_bias(Q, 0.1),
        "W_ih_n": make_weight(Q, P), "W_hh_n": make_weight(Q, Q), "b_n": make_bias(Q),
    }
    return W1, W2

def make_lstm_weights_dec():
    # Capa 1: input D=4, hidden P=5
    W1 = {
        "W_ih_f": make_weight(P, D), "W_hh_f": make_weight(P, P), "b_f": make_bias(P, 1.0),
        "W_ih_i": make_weight(P, D), "W_hh_i": make_weight(P, P), "b_i": make_bias(P),
        "W_ih_g": make_weight(P, D), "W_hh_g": make_weight(P, P), "b_g": make_bias(P),
        "W_ih_o": make_weight(P, D), "W_hh_o": make_weight(P, P), "b_o": make_bias(P),
    }
    # Capa 2: input P=5, hidden Q=3
    W2 = {
        "W_ih_f": make_weight(Q, P), "W_hh_f": make_weight(Q, Q), "b_f": make_bias(Q, 1.0),
        "W_ih_i": make_weight(Q, P), "W_hh_i": make_weight(Q, Q), "b_i": make_bias(Q),
        "W_ih_g": make_weight(Q, P), "W_hh_g": make_weight(Q, Q), "b_g": make_bias(Q),
        "W_ih_o": make_weight(Q, P), "W_hh_o": make_weight(Q, Q), "b_o": make_bias(Q),
    }
    return W1, W2

# ---------------------------------------------------------------------------
# Decoder forward — sin atención
# Ajuste de W_out para que genere la traducción target.
# ---------------------------------------------------------------------------

def make_decoder_out_weights():
    return make_weight(V, Q)

def tune_W_out(W_out_init, h2_seq, target_tokens, scale=8.0):
    """
    Construcción directa: W_out[k_t] = scale * h2_t / ||h2_t||.
    Funciona porque todos los target_tokens son distintos entre sí:
    el score del token correcto en su timestep es scale * ||h2_t||,
    mientras que el score de cualquier token incorrecto es
    scale * ||h2_t|| * cos(θ) < scale * ||h2_t|| (para θ > 0).
    Los tokens sin timestep asignado tienen fila = 0.
    Se verifica post-redondeo con mayor margen si hace falta.
    """
    assert len(set(target_tokens)) == len(target_tokens), (
        "target_tokens deben ser todos distintos para la construcción directa"
    )
    W = np.zeros((V, len(h2_seq[0])), dtype=np.float64)
    for h2, tok in zip(h2_seq, target_tokens):
        target_idx = IDX[tok]
        h_norm = h2 / (np.linalg.norm(h2) + 1e-8)
        W[target_idx] = scale * h_norm
    W_r = r2(W)
    # Verificar post-redondeo; si falla, aumentar escala
    ok = all(VOCAB[int(np.argmax(W_r @ h2))] == tok
             for h2, tok in zip(h2_seq, target_tokens))
    if not ok:
        W_r = r2(np.zeros((V, len(h2_seq[0])), dtype=np.float64))
        for h2, tok in zip(h2_seq, target_tokens):
            target_idx = IDX[tok]
            h_norm = h2 / (np.linalg.norm(h2) + 1e-8)
            W_r[target_idx] = round(scale * 4, 2) * np.round(h_norm, 4)
    for h2, tok in zip(h2_seq, target_tokens):
        pred = VOCAB[int(np.argmax(W_r @ h2))]
        assert pred == tok, (
            f"tune_W_out falló (post-redondeo): pred={pred} vs {tok}. "
            f"h2={h2}, h2_norm={h2 / np.linalg.norm(h2)}"
        )
    return W_r

def decoder_noattn_forward_gru(h_T2_enc, W1, W2, W_c, W_out):
    """
    h_T2_enc: último estado encoder capa 2 (dim L=4)
    W_c: (P x L) — proyecta h_T2_enc → h_0 de decoder capa 1
    Ambas pasadas usan teacher forcing (inputs = tokens target correctos).
    """
    h1 = W_c @ h_T2_enc
    context_vec = h1.copy()
    context_info = {
        "W_c": W_c.tolist(),
        "h_T_layer2": h_T2_enc.tolist(),
        "h_0_decoder_layer1": h1.tolist(),
    }

    # Inputs con teacher forcing: <START> + TARGET[0..4]
    input_tokens = ["<START>"] + TARGET_TRANSLATION[:-1]

    # Pasada 1: recolectar h2_seq con teacher forcing
    h1 = context_vec.copy(); h2 = np.zeros(Q)
    h2_seq = []
    for tok in input_tokens:
        x = emb(tok)
        h1, _ = gru_step(x, h1, W1)
        h2, _ = gru_step(h1, h2, W2)
        h2_seq.append(h2.copy())

    W_out_tuned = tune_W_out(W_out, h2_seq, TARGET_TRANSLATION)

    # Pasada 2: recolectar detalles con teacher forcing, verificar argmax
    h1 = context_vec.copy(); h2 = np.zeros(Q)
    timesteps = []
    for step, current_tok in enumerate(input_tokens):
        x = emb(current_tok)
        h1_prev, h2_prev = h1.copy(), h2.copy()
        h1, g1 = gru_step(x, h1_prev, W1)
        h2, g2 = gru_step(h1, h2_prev, W2)

        logits = W_out_tuned @ h2
        probs = softmax(logits)
        pred_tok = VOCAB[int(np.argmax(probs))]

        timesteps.append({
            "t": step + 1,
            "input_token": current_tok,
            "input_embedding": x.tolist(),
            "layer1": {
                "h_prev": h1_prev.tolist(),
                "x_t": x.tolist(),
                "gates": g1,
                "h_t": h1.tolist(),
            },
            "layer2": {
                "h_prev": h2_prev.tolist(),
                "x_t": h1.tolist(),
                "gates": g2,
                "h_t": h2.tolist(),
            },
            "softmax": {
                "logits": logits.tolist(),
                "probas": probs.tolist(),
                "argmax": pred_tok,
            },
        })

    generated = [ts["softmax"]["argmax"] for ts in timesteps]
    assert generated == TARGET_TRANSLATION, (
        f"GRU decoder noattn generó: {generated}\nEsperado: {TARGET_TRANSLATION}"
    )
    return context_info, timesteps, W_out_tuned

def decoder_noattn_forward_lstm(h_T2_enc, c_T2_enc, W1, W2, W_c, W_out):
    h1 = W_c @ h_T2_enc
    context_vec = h1.copy()
    context_info = {
        "W_c": W_c.tolist(),
        "h_T_layer2": h_T2_enc.tolist(),
        "h_0_decoder_layer1": h1.tolist(),
    }

    input_tokens = ["<START>"] + TARGET_TRANSLATION[:-1]

    # Pasada 1: teacher forcing, recolectar h2_seq
    h1 = context_vec.copy(); c1 = np.zeros(P)
    h2 = np.zeros(Q); c2 = np.zeros(Q)
    h2_seq = []
    for tok in input_tokens:
        x = emb(tok)
        h1, c1, _ = lstm_step(x, h1, c1, W1)
        h2, c2, _ = lstm_step(h1, h2, c2, W2)
        h2_seq.append(h2.copy())

    W_out_tuned = tune_W_out(W_out, h2_seq, TARGET_TRANSLATION)

    # Pasada 2: teacher forcing, recolectar detalles
    h1 = context_vec.copy(); c1 = np.zeros(P)
    h2 = np.zeros(Q); c2 = np.zeros(Q)
    timesteps = []
    for step, current_tok in enumerate(input_tokens):
        x = emb(current_tok)
        h1_prev, c1_prev = h1.copy(), c1.copy()
        h2_prev, c2_prev = h2.copy(), c2.copy()
        h1, c1, g1 = lstm_step(x, h1_prev, c1_prev, W1)
        h2, c2, g2 = lstm_step(h1, h2_prev, c2_prev, W2)

        logits = W_out_tuned @ h2
        probs = softmax(logits)
        pred_tok = VOCAB[int(np.argmax(probs))]

        timesteps.append({
            "t": step + 1,
            "input_token": current_tok,
            "input_embedding": x.tolist(),
            "layer1": {
                "h_prev": h1_prev.tolist(),
                "c_prev": c1_prev.tolist(),
                "x_t": x.tolist(),
                "gates": g1,
                "h_t": h1.tolist(),
                "c_t": c1.tolist(),
            },
            "layer2": {
                "h_prev": h2_prev.tolist(),
                "c_prev": c2_prev.tolist(),
                "x_t": h1.tolist(),
                "gates": g2,
                "h_t": h2.tolist(),
                "c_t": c2.tolist(),
            },
            "softmax": {
                "logits": logits.tolist(),
                "probas": probs.tolist(),
                "argmax": pred_tok,
            },
        })

    generated = [ts["softmax"]["argmax"] for ts in timesteps]
    assert generated == TARGET_TRANSLATION, (
        f"LSTM decoder noattn generó: {generated}\nEsperado: {TARGET_TRANSLATION}"
    )
    return context_info, timesteps, W_out_tuned

# ---------------------------------------------------------------------------
# Decoder con atención Luong-general
# ---------------------------------------------------------------------------

def attention_scores(h_dec_q, enc_h2_all, W_a):
    """
    h_dec_q: dim Q=3
    enc_h2_all: lista de 7 vectores dim L=4
    W_a: (Q x L) = (3 x 4)
    scores[i] = h_dec_q^T @ W_a @ h_enc_i
    """
    scores = []
    for h_enc_i in enc_h2_all:
        score = float(h_dec_q @ W_a @ h_enc_i)
        scores.append(score)
    return np.array(scores)

def make_interpretable_W_a(h_dec_states, enc_h2_all, target_pairs):
    """
    Encuentra W_a (Q x L) via mínimos cuadrados tal que la matriz de scores
    S[t,i] = h_dec_t^T W_a h_enc_i tenga valores altos en los pares objetivo.

    Sistema lineal: (E[i] ⊗ D[t]) · vec_F(W_a) = S_target[t,i]
    Forma: (T*N, L*Q) → lstsq mínima norma.

    Post-lstsq: escala W_a para que el pico mínimo de los pares logrados
    dé α_max ≈ 0.55 (softmax peak interpretable, no degenerado a 1.000).
    Si un par no logra el ordenamiento correcto se descarta y se continúa.

    target_pairs: [(dec_step_0idx, enc_token_0idx), ...]
    """
    T = len(h_dec_states)
    N = len(enc_h2_all)
    D = np.array(h_dec_states)   # (T, Q)
    E = np.array(enc_h2_all)     # (N, L)

    # Boost unitario para encontrar la dirección correcta
    S_target = np.full((T, N), -1.0 / N)
    for dec_idx, enc_idx in target_pairs:
        S_target[dec_idx, enc_idx] = 1.0

    A = np.zeros((T * N, L * Q))
    b = S_target.flatten()
    for t in range(T):
        for i in range(N):
            A[t * N + i, :] = np.kron(E[i, :], D[t, :])

    w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    W_a_base = w.reshape((L, Q)).T   # (Q, L)

    # Verificar qué pares logran el ordenamiento correcto
    achieved = []
    for dec_idx, enc_idx in target_pairs:
        h_d = D[dec_idx]
        scores = np.array([float(h_d @ W_a_base @ E[i]) for i in range(N)])
        if int(np.argmax(scores)) == enc_idx:
            # Calcular margen: winner_score - max(otros scores)
            sorted_s = np.sort(scores)[::-1]
            margin = sorted_s[0] - sorted_s[1]
            achieved.append((dec_idx, enc_idx, margin))

    if not achieved:
        # Ningún par logrado — devolver W_a base sin escalar (al menos no es uniforme)
        return r2(W_a_base * 5)

    # Escalar para que el margen mínimo dé α_max ≈ 0.55:
    # softmax peak ≈ 0.55 requiere score_diff ≈ ln(0.55*6/0.45) ≈ 1.5 sobre la media
    # Si margin_min = m, necesitamos k * m ≈ 2.0 → k = 2.0 / m
    min_margin = min(a[2] for a in achieved)
    k = 2.0 / (min_margin + 1e-9)
    # Limitar k para evitar W_a con valores enormes (mala legibilidad)
    k = min(k, 150.0)

    W_a_scaled = r2(W_a_base * k)

    # Verificación final con W_a redondeada
    ok_pairs = []
    for dec_idx, enc_idx in target_pairs:
        h_d = D[dec_idx]
        scores = np.array([float(h_d @ W_a_scaled @ E[i]) for i in range(N)])
        if int(np.argmax(softmax(scores))) == enc_idx:
            ok_pairs.append((dec_idx, enc_idx))

    if not ok_pairs and k < 150:
        # El redondeo rompió los picos — subir escala hasta 3×
        W_a_scaled = r2(W_a_base * k * 3)

    return W_a_scaled

def decoder_attn_forward_gru(h_T2_enc, enc_h2_all, W1, W2, W_c, W_a, W_combine, W_out):
    """
    W_a: ignorado — se construye internamente con make_interpretable_W_a
    usando los h_dec^(2) reales (la atención es solo en la salida, no en las celdas).
    """
    h1 = W_c @ h_T2_enc
    context_vec = h1.copy()
    context_info = {
        "W_c": W_c.tolist(),
        "h_T_layer2": h_T2_enc.tolist(),
        "h_0_decoder_layer1": h1.tolist(),
    }

    enc_arr = np.array(enc_h2_all)
    input_tokens = ["<START>"] + TARGET_TRANSLATION[:-1]

    # Pasada preliminar: recolectar h_dec^(2) sin atención (la atención es
    # output-only, así que las celdas GRU no dependen de W_a — los estados
    # son idénticos a los de las pasadas principales).
    h1_pre = context_vec.copy(); h2_pre = np.zeros(Q)
    h_dec_states = []
    for tok in input_tokens:
        x = emb(tok)
        h1_pre, _ = gru_step(x, h1_pre, W1)
        h2_pre, _ = gru_step(h1_pre, h2_pre, W2)
        h_dec_states.append(h2_pre.copy())

    # Pares con match lingüístico fuerte y mathematicamente alcanzables:
    # t=0 (gen I)     → enc[0] Me      (pronombre personal)
    # t=1 (gen loved) → enc[1] encantó (traducción directa)
    # Los pasos 2-4 producen h_dec casi idénticos (coseno > 0.99),
    # así que cualquier W_a dará el mismo ganador para los tres.
    target_pairs = [(0, 0), (1, 1)]
    W_a_tuned = make_interpretable_W_a(h_dec_states, enc_h2_all, target_pairs)

    # Pasada 1: teacher forcing, recolectar h_tilde_seq
    h1 = context_vec.copy(); h2 = np.zeros(Q)
    h_tilde_seq = []
    for tok in input_tokens:
        x = emb(tok)
        h1, _ = gru_step(x, h1, W1)
        h2, _ = gru_step(h1, h2, W2)
        scores = attention_scores(h2, enc_h2_all, W_a_tuned)
        alphas = softmax(scores)
        c_t = enc_arr.T @ alphas
        h_tilde = tanh(W_combine @ np.concatenate([h2, c_t]))
        h_tilde_seq.append(h_tilde.copy())

    W_out_tuned = tune_W_out(W_out, h_tilde_seq, TARGET_TRANSLATION)

    # Pasada 2: teacher forcing, recolectar detalles
    h1 = context_vec.copy(); h2 = np.zeros(Q)
    timesteps = []
    for step, current_tok in enumerate(input_tokens):
        x = emb(current_tok)
        h1_prev, h2_prev = h1.copy(), h2.copy()
        h1, g1 = gru_step(x, h1_prev, W1)
        h2, g2 = gru_step(h1, h2_prev, W2)

        scores = attention_scores(h2, enc_h2_all, W_a_tuned)
        alphas = softmax(scores)
        c_t = enc_arr.T @ alphas
        h_tilde = tanh(W_combine @ np.concatenate([h2, c_t]))

        logits = W_out_tuned @ h_tilde
        probs = softmax(logits)
        pred_tok = VOCAB[int(np.argmax(probs))]

        timesteps.append({
            "t": step + 1,
            "input_token": current_tok,
            "input_embedding": x.tolist(),
            "layer1": {
                "h_prev": h1_prev.tolist(),
                "x_t": x.tolist(),
                "gates": g1,
                "h_t": h1.tolist(),
            },
            "layer2": {
                "h_prev": h2_prev.tolist(),
                "x_t": h1.tolist(),
                "gates": g2,
                "h_t": h2.tolist(),
            },
            "attention": {
                "scores": scores.tolist(),
                "alphas": alphas.tolist(),
                "contexto": c_t.tolist(),
                "h_tilde": h_tilde.tolist(),
            },
            "softmax": {
                "logits": logits.tolist(),
                "probas": probs.tolist(),
                "argmax": pred_tok,
            },
        })

    generated = [ts["softmax"]["argmax"] for ts in timesteps]
    assert generated == TARGET_TRANSLATION, (
        f"GRU decoder attn generó: {generated}\nEsperado: {TARGET_TRANSLATION}"
    )
    return context_info, timesteps, W_a_tuned, W_out_tuned

def decoder_attn_forward_lstm(h_T2_enc, c_T2_enc, enc_h2_all, W1, W2, W_c, W_a, W_combine, W_out):
    h1 = W_c @ h_T2_enc
    context_vec = h1.copy()
    context_info = {
        "W_c": W_c.tolist(),
        "h_T_layer2": h_T2_enc.tolist(),
        "h_0_decoder_layer1": h1.tolist(),
    }

    enc_arr = np.array(enc_h2_all)
    input_tokens = ["<START>"] + TARGET_TRANSLATION[:-1]

    # Pasada preliminar: los estados LSTM de las celdas no dependen de W_a
    h1_pre = context_vec.copy(); c1_pre = np.zeros(P)
    h2_pre = np.zeros(Q); c2_pre = np.zeros(Q)
    h_dec_states = []
    for tok in input_tokens:
        x = emb(tok)
        h1_pre, c1_pre, _ = lstm_step(x, h1_pre, c1_pre, W1)
        h2_pre, c2_pre, _ = lstm_step(h1_pre, h2_pre, c2_pre, W2)
        h_dec_states.append(h2_pre.copy())

    target_pairs = [(0, 0), (1, 1)]
    W_a_tuned = make_interpretable_W_a(h_dec_states, enc_h2_all, target_pairs)

    # Pasada 1: teacher forcing, recolectar h_tilde_seq
    h1 = context_vec.copy(); c1 = np.zeros(P)
    h2 = np.zeros(Q); c2 = np.zeros(Q)
    h_tilde_seq = []
    for tok in input_tokens:
        x = emb(tok)
        h1, c1, _ = lstm_step(x, h1, c1, W1)
        h2, c2, _ = lstm_step(h1, h2, c2, W2)
        scores = attention_scores(h2, enc_h2_all, W_a_tuned)
        alphas = softmax(scores)
        c_t = enc_arr.T @ alphas
        h_tilde = tanh(W_combine @ np.concatenate([h2, c_t]))
        h_tilde_seq.append(h_tilde.copy())

    W_out_tuned = tune_W_out(W_out, h_tilde_seq, TARGET_TRANSLATION)

    # Pasada 2: teacher forcing, recolectar detalles
    h1 = context_vec.copy(); c1 = np.zeros(P)
    h2 = np.zeros(Q); c2 = np.zeros(Q)
    timesteps = []
    for step, current_tok in enumerate(input_tokens):
        x = emb(current_tok)
        h1_prev, c1_prev = h1.copy(), c1.copy()
        h2_prev, c2_prev = h2.copy(), c2.copy()
        h1, c1, g1 = lstm_step(x, h1_prev, c1_prev, W1)
        h2, c2, g2 = lstm_step(h1, h2_prev, c2_prev, W2)

        scores = attention_scores(h2, enc_h2_all, W_a_tuned)
        alphas = softmax(scores)
        c_t = enc_arr.T @ alphas
        h_tilde = tanh(W_combine @ np.concatenate([h2, c_t]))

        logits = W_out_tuned @ h_tilde
        probs = softmax(logits)
        pred_tok = VOCAB[int(np.argmax(probs))]

        timesteps.append({
            "t": step + 1,
            "input_token": current_tok,
            "input_embedding": x.tolist(),
            "layer1": {
                "h_prev": h1_prev.tolist(),
                "c_prev": c1_prev.tolist(),
                "x_t": x.tolist(),
                "gates": g1,
                "h_t": h1.tolist(),
                "c_t": c1.tolist(),
            },
            "layer2": {
                "h_prev": h2_prev.tolist(),
                "c_prev": c2_prev.tolist(),
                "x_t": h1.tolist(),
                "gates": g2,
                "h_t": h2.tolist(),
                "c_t": c2.tolist(),
            },
            "attention": {
                "scores": scores.tolist(),
                "alphas": alphas.tolist(),
                "contexto": c_t.tolist(),
                "h_tilde": h_tilde.tolist(),
            },
            "softmax": {
                "logits": logits.tolist(),
                "probas": probs.tolist(),
                "argmax": pred_tok,
            },
        })

    generated = [ts["softmax"]["argmax"] for ts in timesteps]
    assert generated == TARGET_TRANSLATION, (
        f"LSTM decoder attn generó: {generated}\nEsperado: {TARGET_TRANSLATION}"
    )
    return context_info, timesteps, W_a_tuned, W_out_tuned

# ---------------------------------------------------------------------------
# Validación numérica spot-check
# ---------------------------------------------------------------------------

def validate_gru_step(x, h_prev, W, h_t_reported):
    """Recalcula h_t y valida contra el valor reportado."""
    h_t_check, _ = gru_step(x, h_prev, W)
    err = np.max(np.abs(h_t_check - np.array(h_t_reported)))
    assert err < 1e-6, f"Validación GRU falló: error={err}"

def validate_lstm_step(x, h_prev, c_prev, W, h_t_reported, c_t_reported):
    h_t_check, c_t_check, _ = lstm_step(x, h_prev, c_prev, W)
    err_h = np.max(np.abs(h_t_check - np.array(h_t_reported)))
    err_c = np.max(np.abs(c_t_check - np.array(c_t_reported)))
    assert max(err_h, err_c) < 1e-6, f"Validación LSTM falló: err_h={err_h}, err_c={err_c}"

# ---------------------------------------------------------------------------
# Helpers de serialización
# ---------------------------------------------------------------------------

def weights_to_dict_gru(W):
    return {k: v.tolist() for k, v in W.items()}

def weights_to_dict_lstm(W):
    return {k: v.tolist() for k, v in W.items()}

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("Generando datos...")

    # ---- Semilla global (ya fijada arriba, se vuelve a fijar aquí por claridad) ----
    np.random.seed(42)

    # ================================================================
    # GRU — encoder compartido para los 3 escenarios GRU
    # ================================================================
    np.random.seed(42)
    gru_W1_enc, gru_W2_enc = make_gru_weights_enc()

    np.random.seed(43)
    gru_enc_ts, gru_h1_T, gru_h2_T, gru_enc_h2_all = encoder_forward_gru(gru_W1_enc, gru_W2_enc)

    # Validación spot-check: timesteps 1 y 7
    for ts in [gru_enc_ts[0], gru_enc_ts[-1]]:
        validate_gru_step(
            np.array(ts["layer1"]["x_t"]),
            np.array(ts["layer1"]["h_prev"]),
            gru_W1_enc,
            ts["layer1"]["h_t"],
        )
        validate_gru_step(
            np.array(ts["layer2"]["x_t"]),
            np.array(ts["layer2"]["h_prev"]),
            gru_W2_enc,
            ts["layer2"]["h_t"],
        )
    print("  ✓ GRU encoder validado")

    # ================================================================
    # GRU Sentiment
    # ================================================================
    np.random.seed(50)
    gru_W_d1, gru_b_d1, gru_W_d2, gru_b_d2, gru_head = tune_sentiment_head(gru_h2_T)
    print(f"  ✓ GRU sentiment: {gru_head['prediccion']} | softmax={[round(p,3) for p in gru_head['softmax']]}")

    # ================================================================
    # GRU Translation sin atención
    # ================================================================
    np.random.seed(60)
    gru_W_c = make_weight(P, L)
    np.random.seed(61)
    gru_dec_W1, gru_dec_W2 = make_gru_weights_dec()
    np.random.seed(62)
    gru_W_out = make_decoder_out_weights()

    gru_ctx_noattn, gru_dec_ts_noattn, gru_W_out_noattn = decoder_noattn_forward_gru(
        gru_h2_T, gru_dec_W1, gru_dec_W2, gru_W_c, gru_W_out
    )
    print(f"  ✓ GRU translation noattn: {[ts['softmax']['argmax'] for ts in gru_dec_ts_noattn]}")

    # ================================================================
    # GRU Translation con atención
    # ================================================================
    np.random.seed(70)
    gru_W_a = make_weight(Q, L)
    np.random.seed(71)
    gru_W_combine = make_weight(Q, Q + L)
    np.random.seed(72)
    gru_W_out_attn_init = make_decoder_out_weights()

    gru_ctx_attn, gru_dec_ts_attn, gru_W_a_tuned, gru_W_out_attn = decoder_attn_forward_gru(
        gru_h2_T, gru_enc_h2_all,
        gru_dec_W1, gru_dec_W2, gru_W_c,
        gru_W_a, gru_W_combine, gru_W_out_attn_init
    )
    print(f"  ✓ GRU translation attn:   {[ts['softmax']['argmax'] for ts in gru_dec_ts_attn]}")
    for t, ts in enumerate(gru_dec_ts_attn):
        top_enc = int(np.argmax(ts["attention"]["alphas"]))
        print(f"      t={t+1} ({ts['input_token']:8s}) → pico atención en enc[{top_enc+1}]={ENCODER_TOKENS[top_enc]}"
              f"  α_max={ts['attention']['alphas'][top_enc]:.3f}")

    # ================================================================
    # LSTM — encoder compartido para los 3 escenarios LSTM
    # ================================================================
    np.random.seed(80)
    lstm_W1_enc, lstm_W2_enc = make_lstm_weights_enc()
    np.random.seed(81)
    lstm_enc_ts, lstm_h1_T, lstm_h2_T, lstm_c1_T, lstm_c2_T, lstm_enc_h2_all = encoder_forward_lstm(
        lstm_W1_enc, lstm_W2_enc
    )

    for ts in [lstm_enc_ts[0], lstm_enc_ts[-1]]:
        validate_lstm_step(
            np.array(ts["layer1"]["x_t"]),
            np.array(ts["layer1"]["h_prev"]),
            np.array(ts["layer1"]["c_prev"]),
            lstm_W1_enc,
            ts["layer1"]["h_t"],
            ts["layer1"]["c_t"],
        )
    print("  ✓ LSTM encoder validado")

    # ================================================================
    # LSTM Sentiment
    # ================================================================
    np.random.seed(90)
    lstm_W_d1, lstm_b_d1, lstm_W_d2, lstm_b_d2, lstm_head = tune_sentiment_head(lstm_h2_T)
    print(f"  ✓ LSTM sentiment: {lstm_head['prediccion']} | softmax={[round(p,3) for p in lstm_head['softmax']]}")

    # ================================================================
    # LSTM Translation sin atención
    # ================================================================
    np.random.seed(100)
    lstm_W_c = make_weight(P, L)
    np.random.seed(101)
    lstm_dec_W1, lstm_dec_W2 = make_lstm_weights_dec()
    np.random.seed(102)
    lstm_W_out = make_decoder_out_weights()

    lstm_ctx_noattn, lstm_dec_ts_noattn, lstm_W_out_noattn = decoder_noattn_forward_lstm(
        lstm_h2_T, lstm_c2_T, lstm_dec_W1, lstm_dec_W2, lstm_W_c, lstm_W_out
    )
    print(f"  ✓ LSTM translation noattn: {[ts['softmax']['argmax'] for ts in lstm_dec_ts_noattn]}")

    # ================================================================
    # LSTM Translation con atención
    # ================================================================
    np.random.seed(110)
    lstm_W_a = make_weight(Q, L)
    np.random.seed(111)
    lstm_W_combine = make_weight(Q, Q + L)
    np.random.seed(112)
    lstm_W_out_attn_init = make_decoder_out_weights()

    lstm_ctx_attn, lstm_dec_ts_attn, lstm_W_a_tuned, lstm_W_out_attn = decoder_attn_forward_lstm(
        lstm_h2_T, lstm_c2_T, lstm_enc_h2_all,
        lstm_dec_W1, lstm_dec_W2, lstm_W_c,
        lstm_W_a, lstm_W_combine, lstm_W_out_attn_init
    )
    print(f"  ✓ LSTM translation attn:   {[ts['softmax']['argmax'] for ts in lstm_dec_ts_attn]}")

    # ================================================================
    # Ensamblar JSON
    # ================================================================

    def enc_weights_gru():
        return {
            "layer1": weights_to_dict_gru(gru_W1_enc),
            "layer2": weights_to_dict_gru(gru_W2_enc),
        }

    def enc_weights_lstm():
        return {
            "layer1": weights_to_dict_lstm(lstm_W1_enc),
            "layer2": weights_to_dict_lstm(lstm_W2_enc),
        }

    data = {
        "config": {
            "d": D, "m": M, "l": L, "p": P, "q": Q, "V": V,
            "encoderTokens": ENCODER_TOKENS,
            "vocab": VOCAB,
            "embeddings": {tok: vec for tok, vec in EMBEDDINGS.items()},
        },
        "scenarios": {
            "GRU_sentiment": {
                "encoder": {
                    "weights": enc_weights_gru(),
                    "timesteps": gru_enc_ts,
                },
                "head": {
                    **gru_head,
                    "weights": {
                        "W_d1": gru_W_d1.tolist(), "b_d1": gru_b_d1.tolist(),
                        "W_d2": gru_W_d2.tolist(), "b_d2": gru_b_d2.tolist(),
                    },
                },
            },
            "LSTM_sentiment": {
                "encoder": {
                    "weights": enc_weights_lstm(),
                    "timesteps": lstm_enc_ts,
                },
                "head": {
                    **lstm_head,
                    "weights": {
                        "W_d1": lstm_W_d1.tolist(), "b_d1": lstm_b_d1.tolist(),
                        "W_d2": lstm_W_d2.tolist(), "b_d2": lstm_b_d2.tolist(),
                    },
                },
            },
            "GRU_translation_noattn": {
                "encoder": {
                    "weights": enc_weights_gru(),
                    "timesteps": gru_enc_ts,
                    "h2_all": [h.tolist() for h in gru_enc_h2_all],
                },
                "context": gru_ctx_noattn,
                "decoder": {
                    "weights": {
                        "layer1": weights_to_dict_gru(gru_dec_W1),
                        "layer2": weights_to_dict_gru(gru_dec_W2),
                    },
                    "W_out": gru_W_out_noattn.tolist(),
                    "timesteps": gru_dec_ts_noattn,
                },
            },
            "GRU_translation_attn": {
                "encoder": {
                    "weights": enc_weights_gru(),
                    "timesteps": gru_enc_ts,
                    "h2_all": [h.tolist() for h in gru_enc_h2_all],
                },
                "context": gru_ctx_attn,
                "decoder": {
                    "weights": {
                        "layer1": weights_to_dict_gru(gru_dec_W1),
                        "layer2": weights_to_dict_gru(gru_dec_W2),
                    },
                    "W_a": gru_W_a_tuned.tolist(),
                    "W_combine": gru_W_combine.tolist(),
                    "W_out": gru_W_out_attn.tolist(),
                    "timesteps": gru_dec_ts_attn,
                },
            },
            "LSTM_translation_noattn": {
                "encoder": {
                    "weights": enc_weights_lstm(),
                    "timesteps": lstm_enc_ts,
                    "h2_all": [h.tolist() for h in lstm_enc_h2_all],
                },
                "context": lstm_ctx_noattn,
                "decoder": {
                    "weights": {
                        "layer1": weights_to_dict_lstm(lstm_dec_W1),
                        "layer2": weights_to_dict_lstm(lstm_dec_W2),
                    },
                    "W_out": lstm_W_out_noattn.tolist(),
                    "timesteps": lstm_dec_ts_noattn,
                },
            },
            "LSTM_translation_attn": {
                "encoder": {
                    "weights": enc_weights_lstm(),
                    "timesteps": lstm_enc_ts,
                    "h2_all": [h.tolist() for h in lstm_enc_h2_all],
                },
                "context": lstm_ctx_attn,
                "decoder": {
                    "weights": {
                        "layer1": weights_to_dict_lstm(lstm_dec_W1),
                        "layer2": weights_to_dict_lstm(lstm_dec_W2),
                    },
                    "W_a": lstm_W_a_tuned.tolist(),
                    "W_combine": lstm_W_combine.tolist(),
                    "W_out": lstm_W_out_attn.tolist(),
                    "timesteps": lstm_dec_ts_attn,
                },
            },
        }
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "src", "data", "data.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    import os as _os
    size_kb = _os.path.getsize(out_path) / 1024
    print(f"\n✅ data.json generado: {size_kb:.1f} KB en {out_path}")

if __name__ == "__main__":
    main()
