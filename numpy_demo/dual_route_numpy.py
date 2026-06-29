"""A self-contained NumPy realization of the dual-route repetition model, with a
**parametric recurrent serial-recall dorsal route** (Botvinick & Plaut, 2006).

Both routes are now learned and trained end-to-end with hand-written backprop:

  * Ventral (LTM) route  : a parametric autoencoder through a tight semantic
    bottleneck (form -> meaning -> form). Memorizes trained words; on novel forms
    it falls back on structure. Lexical, frequency-sensitive.
  * Dorsal (WM) route     : a parametric Elman **encoder -> decoder** that reads the
    phoneme sequence into a bounded recurrent state and re-articulates it. It
    learns the auditory->motor map and phonotactics, generalizes to novel words,
    and its bounded state + interference noise give capacity / length limits and
    the serial-position curve. Sub-lexical, frequency-invariant.
  * Gate + shared output : a lexical-familiarity gate mixes the two routes'
    phoneme distributions.

The dorsal route is the part that changed: it is no longer a non-parametric copy
buffer. Frequency-invariance now emerges because the recurrent route learns the
*general* serial-recall computation (it generalizes) rather than from having no
weights.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Data: encode words as fixed-length one-hot phoneme grids
# ---------------------------------------------------------------------------
def encode_dataset(entries, vocab, L):
    """Return X (N, L, V) one-hot, T (N, L) target ids, M (N, L) valid mask."""
    V = vocab.size
    N = len(entries)
    X = np.zeros((N, L, V), dtype=np.float32)
    T = np.full((N, L), vocab.pad_id, dtype=np.int64)
    M = np.zeros((N, L), dtype=np.float32)
    for i, e in enumerate(entries):
        ph = e.phonemes[:L]
        for t, pid in enumerate(ph):
            X[i, t, pid] = 1.0
            T[i, t] = pid
            M[i, t] = 1.0
        for t in range(len(ph), L):           # pad positions
            X[i, t, vocab.pad_id] = 1.0
    return X, T, M


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _softmax_backward(p, dp):
    """Backprop through softmax given upstream dp on probabilities p."""
    return p * (dp - (dp * p).sum(axis=-1, keepdims=True))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DualRouteNumpy:
    def __init__(self, L, V, bos_id, pad_id, emb=24, d_wm=96, h_sem=256,
                 gate_alpha=40.0, gate_thr=0.90, seed=0):
        self.L, self.V, self.H = L, V, h_sem
        self.d_wm, self.emb = d_wm, emb
        self.bos_id, self.pad_id = bos_id, pad_id
        self.gate_alpha, self.gate_thr = gate_alpha, gate_thr
        rng = np.random.default_rng(seed)

        def n(shape, scale):
            return (rng.normal(size=shape) * scale).astype(np.float32)

        # ---- ventral autoencoder (feedforward) ----
        din = L * V
        self.W1 = n((h_sem, din), 1.0 / np.sqrt(din)); self.b1 = np.zeros(h_sem, np.float32)
        self.W2 = n((din, h_sem), 1.0 / np.sqrt(h_sem)); self.b2 = np.zeros(din, np.float32)

        # ---- dorsal recurrent serial-recall network ----
        self.E = n((V, emb), 0.1)                      # shared phoneme embedding
        self.Wxh = n((d_wm, emb), 1.0 / np.sqrt(emb))  # encoder input
        self.Whh = n((d_wm, d_wm), 1.0 / np.sqrt(d_wm))  # encoder recurrent
        self.bh = np.zeros(d_wm, np.float32)
        self.Wdx = n((d_wm, emb), 1.0 / np.sqrt(emb))  # decoder input (prev phoneme)
        self.Wdh = n((d_wm, d_wm), 1.0 / np.sqrt(d_wm))  # decoder recurrent
        self.bd = np.zeros(d_wm, np.float32)
        self.Wo = n((V, d_wm), 1.0 / np.sqrt(d_wm))    # decoder output
        self.bo = np.zeros(V, np.float32)

        self.fam_bank = None

    PARAMS = ["W1", "b1", "W2", "b2", "E", "Wxh", "Whh", "bh",
              "Wdx", "Wdh", "bd", "Wo", "bo"]

    def get_params(self):
        return {k: getattr(self, k) for k in self.PARAMS}

    def set_params(self, p):
        for k, v in p.items():
            setattr(self, k, v)

    def set_familiarity_bank(self, X_train):
        Xf = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
        self.fam_bank = Xf / (np.linalg.norm(Xf, axis=1, keepdims=True) + 1e-8)

    # ------------------------------------------------------------------ ventral
    def _ventral(self, Xf, v_lesion, rng):
        z1 = Xf @ self.W1.T + self.b1
        h = np.tanh(z1)
        if v_lesion and v_lesion[0] > 0 and rng is not None:
            mask = (rng.random(self.H) >= v_lesion[0]).astype(np.float32)
            h = h * mask[None, :]
        vlog = (h @ self.W2.T + self.b2).reshape(-1, self.L, self.V)
        if v_lesion and v_lesion[1] > 0 and rng is not None:
            vlog = vlog + rng.normal(size=vlog.shape).astype(np.float32) * v_lesion[1]
        return softmax(vlog), h

    # ------------------------------------------------------------------ dorsal
    def _dorsal(self, X, T, M, d_lesion, rng):
        """Recurrent encoder->decoder. Returns pd and a cache for BPTT."""
        B = X.shape[0]
        D = self.d_wm
        Xemb = X @ self.E                                   # (B, L, emb)
        # encoder
        h = np.zeros((B, D), np.float32)
        Hpre, H = [], []
        for t in range(self.L):
            a = Xemb[:, t] @ self.Wxh.T + h @ self.Whh.T + self.bh
            ht = np.tanh(a)
            mt = M[:, t:t + 1]
            h = mt * ht + (1.0 - mt) * h                    # carry over pads
            Hpre.append(ht); H.append(h)
        Hpre = np.stack(Hpre, 1); H = np.stack(H, 1)        # (B, L, D)
        context = H[:, -1]                                  # last real hidden

        # decoder (teacher forced on the gold previous phoneme)
        dec_ids = np.concatenate(
            [np.full((B, 1), self.bos_id, np.int64), T[:, :-1]], axis=1)  # (B, L)
        Demb = self.E[dec_ids]                              # (B, L, emb)
        smask = None
        if d_lesion and d_lesion[0] > 0 and rng is not None:
            smask = (rng.random(D) >= d_lesion[0]).astype(np.float32)
        s = context
        S, logits = [], []
        for i in range(self.L):
            g = Demb[:, i] @ self.Wdx.T + s @ self.Wdh.T + self.bd
            si = np.tanh(g)
            if smask is not None:
                si = si * smask[None, :]
            s = si
            S.append(si)
            logits.append(si @ self.Wo.T + self.bo)
        S = np.stack(S, 1)                                  # (B, L, D)
        dlog = np.stack(logits, 1)                          # (B, L, V)
        if d_lesion and d_lesion[1] > 0 and rng is not None:
            dlog = dlog + rng.normal(size=dlog.shape).astype(np.float32) * d_lesion[1]
        cache = dict(Xemb=Xemb, Hpre=Hpre, H=H, context=context, S=S,
                     Demb=Demb, dec_ids=dec_ids, enc_ids=T, M=M, smask=smask)
        return softmax(dlog), cache

    # ------------------------------------------------------------------ forward
    def forward(self, X, T, M, lesion=None, rng=None):
        lesion = lesion or {}
        Xf = X.reshape(X.shape[0], -1)
        pv, hsem = self._ventral(Xf, lesion.get("ventral"), rng)
        pd, dcache = self._dorsal(X, T, M, lesion.get("dorsal"), rng)
        # error-suppression gate from lexical familiarity (input -> known words)
        if self.fam_bank is not None:
            xn = Xf / (np.linalg.norm(Xf, axis=1, keepdims=True) + 1e-8)
            conf = (xn @ self.fam_bank.T).max(axis=1)
        else:
            conf = (M * pv.max(2)).sum(1) / np.maximum(M.sum(1), 1)
        g = 1.0 / (1.0 + np.exp(-self.gate_alpha * (conf - self.gate_thr)))
        gB = g[:, None, None]
        p = gB * pv + (1.0 - gB) * pd
        cache = dict(Xf=Xf, hsem=hsem, pv=pv, pd=pd, g=g, conf=conf, p=p, d=dcache)
        return p, cache

    # ------------------------------------------------------------- loss + grads
    def loss_and_grad(self, X, T, M, lambda_v=1.0, lambda_d=1.0):
        B = X.shape[0]
        p, c = self.forward(X, T, M)
        pv, pd, g, hsem = c["pv"], c["pd"], c["g"], c["hsem"]
        eps = 1e-9
        N = M.sum()
        idx = np.indices((B, self.L))
        pt = p[idx[0], idx[1], T]
        pvt = pv[idx[0], idx[1], T]
        pdt = pd[idx[0], idx[1], T]
        loss = (-(M * np.log(pt + eps)).sum()
                - lambda_v * (M * np.log(pvt + eps)).sum()
                - lambda_d * (M * np.log(pdt + eps)).sum()) / N

        dp = np.zeros_like(p); dp[idx[0], idx[1], T] = -(M / (pt + eps)) / N
        dpv_aux = np.zeros_like(pv); dpv_aux[idx[0], idx[1], T] = -(M / (pvt + eps)) / N * lambda_v
        dpd_aux = np.zeros_like(pd); dpd_aux[idx[0], idx[1], T] = -(M / (pdt + eps)) / N * lambda_d
        gB = g[:, None, None]

        # ---- ventral grads (feedforward) ----
        dvlog = _softmax_backward(pv, gB * dp) + _softmax_backward(pv, dpv_aux)
        dvlog_f = dvlog.reshape(B, -1)
        grads = {}
        grads["W2"] = dvlog_f.T @ hsem
        grads["b2"] = dvlog_f.sum(0)
        dh = dvlog_f @ self.W2
        dz1 = dh * (1 - hsem ** 2)
        grads["W1"] = dz1.T @ c["Xf"]
        grads["b1"] = dz1.sum(0)

        # ---- dorsal grads (BPTT) ----
        ddlog = _softmax_backward(pd, (1.0 - gB) * dp) + _softmax_backward(pd, dpd_aux)
        grads.update(self._dorsal_backward(ddlog, c["d"]))
        return loss, grads

    def dorsal_only_loss_and_grad(self, X, T, M):
        """Train ONLY the dorsal route to repeat a sequence (serial recall).
        Used to expose the buffer to a frequency-flat stream of pronounceable
        forms so it learns a general copy/phonotactic map and stays
        lexical-frequency-invariant. Returns grads for dorsal params only.
        """
        pd, dcache = self._dorsal(X, T, M, None, None)
        B = X.shape[0]
        eps = 1e-9
        N = M.sum()
        idx = np.indices((B, self.L))
        pdt = pd[idx[0], idx[1], T]
        loss = -(M * np.log(pdt + eps)).sum() / N
        dpd = np.zeros_like(pd)
        dpd[idx[0], idx[1], T] = -(M / (pdt + eps)) / N
        ddlog = _softmax_backward(pd, dpd)
        return loss, self._dorsal_backward(ddlog, dcache)

    def _dorsal_backward(self, ddlog, d):
        B, L, D = ddlog.shape[0], self.L, self.d_wm
        Xemb, Hpre, H, context = d["Xemb"], d["Hpre"], d["H"], d["context"]
        S, Demb, dec_ids, enc_ids, M = d["S"], d["Demb"], d["dec_ids"], d["enc_ids"], d["M"]
        smask = d["smask"]
        sm = 1.0 if smask is None else smask[None, :]

        dE = np.zeros_like(self.E)
        dWo = np.zeros_like(self.Wo); dbo = np.zeros_like(self.bo)
        dWdx = np.zeros_like(self.Wdx); dWdh = np.zeros_like(self.Wdh); dbd = np.zeros_like(self.bd)
        dWxh = np.zeros_like(self.Wxh); dWhh = np.zeros_like(self.Whh); dbh = np.zeros_like(self.bh)

        # decoder backward
        ds_next = np.zeros((B, D), np.float32)
        dcontext = np.zeros((B, D), np.float32)
        for i in reversed(range(L)):
            si = S[:, i]
            dlo = ddlog[:, i]                              # (B, V)
            dWo += dlo.T @ si
            dbo += dlo.sum(0)
            dsi = (dlo @ self.Wo + ds_next) * sm           # gate through lesion mask
            da = dsi * (1 - si ** 2)
            s_prev = context if i == 0 else S[:, i - 1]
            dWdh += da.T @ s_prev
            dWdx += da.T @ Demb[:, i]
            dbd += da.sum(0)
            np.add.at(dE, dec_ids[:, i], da @ self.Wdx)
            ds_prev = da @ self.Wdh
            if i == 0:
                dcontext += ds_prev
            else:
                ds_next = ds_prev

        # encoder backward (context = H[:, L-1])
        dh_next = dcontext
        for t in reversed(range(L)):
            mt = M[:, t:t + 1]
            h_prev = H[:, t - 1] if t > 0 else np.zeros((B, D), np.float32)
            da = (dh_next * mt) * (1 - Hpre[:, t] ** 2)
            dWxh += da.T @ Xemb[:, t]
            dWhh += da.T @ h_prev
            dbh += da.sum(0)
            np.add.at(dE, enc_ids[:, t], da @ self.Wxh)
            dh_next = da @ self.Whh + dh_next * (1 - mt)   # tanh path + carry path

        return {"E": dE, "Wxh": dWxh, "Whh": dWhh, "bh": dbh,
                "Wdx": dWdx, "Wdh": dWdh, "bd": dbd, "Wo": dWo, "bo": dbo}


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------
class Adam:
    def __init__(self, params, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (grads[k] ** 2)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = params[k] - self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return params


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def accuracy(model, X, T, M, route="gated", noise_std=0.0, lengths=None,
             rng=None, lesion=None):
    p, c = model.forward(X, T, M, lesion=lesion, rng=rng)
    probs = {"gated": p, "ventral": c["pv"], "dorsal": c["pd"]}[route]
    pred = probs.argmax(-1)
    correct = (pred == T).astype(np.float32) * M
    phon = correct.sum() / M.sum()
    word = ((correct.sum(1)) == M.sum(1)).astype(np.float32).mean()
    return float(phon), float(word)
