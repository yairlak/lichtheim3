"""A self-contained NumPy realization of the dual-route repetition model.

Why this file exists
--------------------
The production model in `models/` is PyTorch. This NumPy version is a compact,
dependency-light twin that *actually trains end-to-end with hand-written
backprop*, so the dual-route idea can be demonstrated (and the loss curve +
unseen-word evaluation produced) in any environment, even without torch.

It keeps the three essential commitments of the full model:

  * Ventral (LTM) route  : a PARAMETRIC autoencoder through a semantic
    bottleneck (form -> meaning -> form). It is trained, so it fits the words it
    has seen but generalizes only partially to novel words.
  * Dorsal (WM) route    : a NON-PARAMETRIC copy buffer. It re-articulates the
    phonemes actually presented (content-agnostic), so it generalizes to unseen
    words for free, but interference noise that scales with list length makes it
    error-prone on long items.
  * Gate + shared output : a learned scalar gate mixes the two streams'
    phoneme distributions, and a single softmax "motor" output articulates.

Evaluation follows the spirit of Chang et al. (arXiv:2506.13450): after
training we probe the model on a disjoint set of UNSEEN words (novel-word /
non-word repetition) and report each route's generalization separately. The
prediction of dual-route theory — and what this demo shows — is a dissociation:
the lexical route's advantage is largely confined to trained words, while the
buffer carries novel words.
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DualRouteNumpy:
    def __init__(self, L, V, h_sem=256, beta=6.0, leak=0.9, k_slots=4,
                 primacy_gain=1.6, primacy_decay=0.18,
                 gate_alpha=10.0, gate_thr=0.5, seed=0):
        self.L, self.V, self.H = L, V, h_sem
        self.beta = beta                       # dorsal copy sharpness
        self.k_slots = k_slots                 # WM capacity (slots)
        # error-suppression gate: trust the lexicon only when it is confident
        self.gate_alpha = gate_alpha
        self.gate_thr = gate_thr
        rng = np.random.default_rng(seed)
        din = L * V
        s = 1.0 / np.sqrt(din)
        # ventral autoencoder params (the only trained weights)
        self.W1 = (rng.normal(size=(h_sem, din)) * s).astype(np.float32)
        self.b1 = np.zeros(h_sem, np.float32)
        self.W2 = (rng.normal(size=(din, h_sem)) * (1/np.sqrt(h_sem))).astype(np.float32)
        self.b2 = np.zeros(din, np.float32)
        # familiarity bank (flattened, L2-normalized training forms); set later.
        # When present, the gate reads lexical FAMILIARITY (max similarity to a
        # known word) instead of raw output confidence -- this is the signal the
        # full model's error_suppression gate gets from its semantic bank, and it
        # is robust to the lexical route being confidently wrong on novel items.
        self.fam_bank = None
        # dorsal trace strength = primacy gradient x recency leak (fixed, content-free)
        pos = np.arange(L)
        recency = leak ** (L - 1 - pos)
        primacy = primacy_gain * np.exp(-primacy_decay * pos)
        self.trace_strength = (primacy * recency).astype(np.float32)   # (L,)
        self.pos_decay = recency.astype(np.float32)

    PARAMS = ["W1", "b1", "W2", "b2"]

    def set_familiarity_bank(self, X_train):
        """Store L2-normalized flattened training forms for the gate."""
        Xf = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
        self.fam_bank = Xf / (np.linalg.norm(Xf, axis=1, keepdims=True) + 1e-8)

    def get_params(self):
        return {k: getattr(self, k) for k in self.PARAMS}

    def set_params(self, p):
        for k, v in p.items():
            setattr(self, k, v)

    # ---- dorsal (non-parametric, capacity-limited) ----
    def dorsal_logits(self, X, M=None, noise_std=0.0, lengths=None, rng=None,
                      lesion_frac=0.0, lesion_noise=0.0):
        """Copy the presented phonemes, but only the `k_slots` strongest
        positions survive the buffer. Dropped positions get no copy (the buffer
        forgot them), so accuracy falls as words exceed capacity -> length effect.

        Ueno-style lesion: `lesion_frac` removes that fraction of the surviving
        copied positions (loss of incoming links to the dorsal/iSMG layer) and
        `lesion_noise` adds Gaussian noise over its output.
        """
        B, L, V = X.shape
        strength = np.tile(self.trace_strength[None, :], (B, 1)).astype(np.float32)
        if M is not None:                       # never keep padding positions
            strength = np.where(M > 0, strength, -1e9)
        # keep the top-k strongest real positions
        keep = min(self.k_slots, L)
        kth = np.sort(strength, axis=1)[:, -keep][:, None]
        kept = (strength >= kth).astype(np.float32)        # (B, L)
        if lesion_frac > 0 and rng is not None:            # remove incoming links
            survive = (rng.random((B, L)) >= lesion_frac).astype(np.float32)
            kept = kept * survive
        dlog = self.beta * X * self.pos_decay[None, :, None] * kept[:, :, None]
        if rng is not None and (noise_std > 0 or lesion_noise > 0):
            # baseline interference scales with overload; lesion noise is flat
            if lengths is not None and noise_std > 0:
                base = noise_std * np.sqrt(np.maximum(lengths, 1) / self.k_slots)
                scale = (base + lesion_noise)[:, None, None]
            else:
                scale = float(noise_std) + float(lesion_noise)
            dlog = dlog + rng.normal(size=dlog.shape).astype(np.float32) * scale
        return dlog

    # ---- forward ----
    def forward(self, X, M=None, noise_std=0.0, lengths=None, rng=None,
                lesion=None):
        """`lesion` is an optional dict like
        {"ventral": (frac, noise), "dorsal": (frac, noise)} simulating Ueno-style
        damage to either pathway (remove a fraction of units/links + output noise).
        """
        lesion = lesion or {}
        v_frac, v_noise = lesion.get("ventral", (0.0, 0.0))
        d_frac, d_noise = lesion.get("dorsal", (0.0, 0.0))
        B = X.shape[0]
        Xf = X.reshape(B, -1)
        z1 = Xf @ self.W1.T + self.b1
        h = np.tanh(z1)
        if v_frac > 0 and rng is not None:          # remove ventral units (cortex)
            mask = (rng.random(self.H) >= v_frac).astype(np.float32)
            h = h * mask[None, :]
        vlog = (h @ self.W2.T + self.b2).reshape(B, self.L, self.V)
        if v_noise > 0 and rng is not None:         # noise over ventral output
            vlog = vlog + rng.normal(size=vlog.shape).astype(np.float32) * v_noise
        pv = softmax(vlog)
        dlog = self.dorsal_logits(X, M=M, noise_std=noise_std,
                                  lengths=lengths, rng=rng,
                                  lesion_frac=d_frac, lesion_noise=d_noise)
        pd = softmax(dlog)
        # error-suppression gate. Familiarity (similarity to a known word) if a
        # bank is available -- robust and generalizes to novel items; otherwise
        # fall back to the ventral route's own output confidence.
        Mc = np.ones((X.shape[0], self.L), np.float32) if M is None else M
        if self.fam_bank is not None:
            xf_n = Xf / (np.linalg.norm(Xf, axis=1, keepdims=True) + 1e-8)
            conf = (xf_n @ self.fam_bank.T).max(axis=1)              # (B,) familiarity
        else:
            conf = (Mc * pv.max(axis=2)).sum(1) / np.maximum(Mc.sum(1), 1)
        g = 1.0 / (1.0 + np.exp(-self.gate_alpha * (conf - self.gate_thr)))
        gB = g[:, None, None]
        p = gB * pv + (1.0 - gB) * pd
        cache = dict(Xf=Xf, h=h, pv=pv, pd=pd, g=g, conf=conf, p=p)
        return p, cache

    # ---- loss + grads ----
    def loss_and_grad(self, X, T, M, lambda_v=1.0, sample_w=None):
        """Gated repetition CE + an auxiliary VENTRAL-only CE (lambda_v).

        The auxiliary term plays the role of the full model's lambda_dec: it
        forces the lexical autoencoder to actually learn to reproduce forms,
        rather than letting the optimiser hide behind the (capacity-limited)
        dorsal copy. The dorsal route has no parameters, so it needs no loss.

        `sample_w` (B,) gives per-word weights -- pass log-frequency weights to
        train on log frequency (frequent words contribute more gradient).
        """
        B = X.shape[0]
        p, c = self.forward(X, M=M)
        h, pv, pd, g = c["h"], c["pv"], c["pd"], c["g"]
        eps = 1e-9
        if sample_w is None:
            Mw = M
        else:
            Mw = M * np.asarray(sample_w, np.float32)[:, None]    # weight rows
        N = Mw.sum()
        idx = np.indices((B, self.L))
        p_true = p[idx[0], idx[1], T]                             # (B, L)
        pv_true = pv[idx[0], idx[1], T]
        loss_rep = -(Mw * np.log(p_true + eps)).sum() / N
        loss_ven = -(Mw * np.log(pv_true + eps)).sum() / N
        loss = loss_rep + lambda_v * loss_ven

        # --- grad of gated repetition term (gate treated as stop-gradient) ---
        dp = np.zeros_like(p)
        dp[idx[0], idx[1], T] = -(Mw / (p_true + eps)) / N
        gB = g[:, None, None]
        dpv = gB * dp
        dvlog = pv * (dpv - (dpv * pv).sum(-1, keepdims=True))

        # --- grad of auxiliary ventral term (straight CE through pv) ---
        dpv_aux = np.zeros_like(pv)
        dpv_aux[idx[0], idx[1], T] = -(Mw / (pv_true + eps)) / N
        dvlog += lambda_v * pv * (dpv_aux - (dpv_aux * pv).sum(-1, keepdims=True))

        dvlog_f = dvlog.reshape(B, -1)
        dW2 = dvlog_f.T @ h
        db2 = dvlog_f.sum(0)
        dh = dvlog_f @ self.W2
        dz1 = dh * (1.0 - h ** 2)
        dW1 = dz1.T @ c["Xf"]
        db1 = dz1.sum(0)
        grads = dict(W1=dW1, b1=db1, W2=dW2, b2=db2)
        return loss, grads


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
    p, c = model.forward(X, M=M, noise_std=noise_std, lengths=lengths, rng=rng,
                         lesion=lesion)
    probs = {"gated": p, "ventral": c["pv"], "dorsal": c["pd"]}[route]
    pred = probs.argmax(-1)
    correct = (pred == T).astype(np.float32) * M
    phon_acc = correct.sum() / M.sum()
    word_correct = ((correct.sum(1)) == M.sum(1)).astype(np.float32)
    word_acc = word_correct.mean()
    return float(phon_acc), float(word_acc)
