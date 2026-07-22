"""Training loop and a `build_and_train` convenience entry point."""
from __future__ import annotations

import itertools
from typing import Tuple

import torch
import torch.nn.functional as F

from config import Config, get_effective_split_seed
from data.phonemes import build_vocab, Vocab
from data.lexicon import build_lexicon, Lexicon
from data.dataset import make_loader, build_pool_loader
from models.dual_route import DualRouteModel
from losses import total_loss
from utils.seed import set_seed


def build_everything(cfg: Config):
    set_seed(cfg.train.seed)
    vocab = build_vocab()
    lexicon = build_lexicon(cfg.data, vocab)
    density = lexicon.neighborhood_density()
    train_entries, val_entries = lexicon.split(cfg.data.val_fraction, get_effective_split_seed(cfg.data))
    num_workers = getattr(cfg.train, "num_workers", 0)

    train_loader = make_loader(
        train_entries, vocab, density, cfg.train.batch_size,
        frequency_weighted=True, freq_temp=cfg.data.freq_temp, shuffle=True,
        num_workers=num_workers)
    val_loader = make_loader(
        val_entries, vocab, density, cfg.train.batch_size,
        frequency_weighted=False, shuffle=False,
        num_workers=num_workers)

    pool_loader = None
    if cfg.train.dorsal_pool_size > 0:
        pool_loader = build_pool_loader(
            vocab, cfg.train.dorsal_pool_size, cfg.train.batch_size,
            cfg.data.semantic_dim, max_len=cfg.data.max_phonemes, seed=cfg.train.seed,
            num_workers=num_workers)

    model = DualRouteModel(cfg, vocab).to(cfg.train.device)
    # frozen lexical bank = GloVe vectors of the *training* lexicon
    bank = torch.stack([torch.tensor(e.semantic) for e in train_entries]).float()
    model.set_semantic_bank(bank.to(cfg.train.device))
    return model, vocab, lexicon, train_loader, val_loader, pool_loader


def _forward_scheduled_sampling(model, batch, tf_ratio: float,
                                pad_id: int, device) -> dict:
    """Encode-once, decode-stepwise scheduled sampling (tf_ratio < 1.0).

    Encodes both routes a single time per batch (drawing all route noises once
    per stimulus), then steps through the decoder autoregressively.

    At each decoder step t:
    - with probability tf_ratio   → feed the gold previous token
    - with probability 1-tf_ratio → feed the model's own argmax prediction

    Returns a dict matching model.forward() in structure, with logits of
    shape (B, S, V) aligned to batch["dec_tgt"].

    Scientific noise semantics
    --------------------------
    With the encode-once design, WM interference noise and LTM ventral noise
    are each sampled ONCE per stimulus per training step (in encode_all), not
    once per decoder step.  This is the intended cognitive interpretation:
    one corruption of the phonological buffer per stimulus presentation.
    """
    enc_in   = batch["enc_in"]
    enc_mask = batch["enc_mask"]
    dec_in   = batch["dec_in"]    # (B, S) = [BOS, p1, ..., pT, PAD...]
    B, S     = dec_in.shape

    # Encode ONCE — noise drawn once per item for this forward pass
    encoded = model.encode_all(enc_in, enc_mask)
    h_WM  = encoded["h_WM"]
    s_hat = encoded["s_hat"]
    field = encoded["field"]

    current_dec = dec_in[:, :1]   # start with BOS only: (B, 1)
    lists = {k: [] for k in ("logits", "wm_logits", "ltm_logits", "gate")}

    for step in range(S):
        out = model.decode_from_states(h_WM, s_hat, field, current_dec)

        # Collect only the last-position logit/gate at each step
        lists["logits"].append(out["logits"][:, -1:, :])       # (B,1,V)
        lists["wm_logits"].append(out["wm_logits"][:, -1:, :])
        lists["ltm_logits"].append(out["ltm_logits"][:, -1:, :])
        lists["gate"].append(out["gate"][:, -1:, :])           # (B,1,1)

        if step < S - 1:
            gold_next = dec_in[:, step + 1: step + 2]          # (B,1) gold
            if tf_ratio <= 0.0:
                next_tok = out["logits"][:, -1, :].argmax(-1, keepdim=True).detach()
            else:
                pred_next = out["logits"][:, -1, :].argmax(-1, keepdim=True).detach()
                use_gold  = torch.rand(B, 1, device=device) < tf_ratio
                next_tok  = torch.where(use_gold, gold_next, pred_next)
            current_dec = torch.cat([current_dec, next_tok], dim=1)

    result = {
        "logits":     torch.cat(lists["logits"],     dim=1),   # (B,S,V)
        "wm_logits":  torch.cat(lists["wm_logits"],  dim=1),
        "ltm_logits": torch.cat(lists["ltm_logits"], dim=1),
        "gate":       torch.cat(lists["gate"],        dim=1),
        "s_hat":      s_hat,
    }
    if field is not None:
        result.update({f"field_{k}": v for k, v in field.items()})
    return result


def run_epoch(model, loader, cfg: Config, optim=None, pool_iter=None) -> dict:
    train_mode = optim is not None
    model.train(train_mode)
    pad_id = model.vocab.pad_id
    dev = cfg.train.device
    tf_ratio = getattr(cfg.train, "teacher_forcing_ratio", 1.0)
    agg = {}
    n = 0
    for batch in loader:
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        if train_mode and tf_ratio < 1.0:
            out = _forward_scheduled_sampling(model, batch, tf_ratio, pad_id, dev)
        else:
            out = model(batch["enc_in"], batch["enc_mask"], batch["dec_in"])
        losses = total_loss(out, batch, cfg.loss, pad_id,
                            usage_prior=cfg.gating.usage_prior)
        total = losses["total"]
        if train_mode and pool_iter is not None:
            # dorsal-only serial-recall loss on a pronounceable pseudoword batch
            pb = next(pool_iter)
            pb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in pb.items()}
            pout = model(pb["enc_in"], pb["enc_mask"], pb["dec_in"])
            V = pout["wm_logits"].shape[-1]
            pool_ce = F.cross_entropy(pout["wm_logits"].reshape(-1, V),
                                      pb["dec_tgt"].reshape(-1), ignore_index=pad_id)
            total = total + cfg.loss.wm * pool_ce
        if train_mode:
            optim.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optim.step()
        for k, v in losses.items():
            agg[k] = agg.get(k, 0.0) + v.detach().item() * len(batch["words"])
        n += len(batch["words"])
    return {k: v / max(n, 1) for k, v in agg.items()}


def plot_loss_history(history, path: str) -> None:
    """Save a train/val loss curve (matplotlib import kept local & optional)."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ep, [h["train_total"] for h in history], label="train total")
    ax.plot(ep, [h["train_rep"] for h in history], label="train rep (motor)")
    ax.plot(ep, [h["val_rep"] for h in history], "--", label="val rep (unseen)")
    ax.set(title="Training loss", xlabel="epoch", ylabel="loss")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def build_and_train(cfg: Config, out_dir: str = None
                    ) -> Tuple[DualRouteModel, Vocab, Lexicon, list,
                               "torch.optim.Optimizer"]:
    """Train model from scratch and return (model, vocab, lexicon, history, optim).

    Returns optimizer as the 5th element so callers can save optimizer_state_dict
    in fresh-mode checkpoints.  Previously, the optimizer was not exposed.

    Backward compat: callers that only unpack 4 values can ignore the 5th:
        model, vocab, lexicon, history, *_ = build_and_train(cfg)
    """
    if cfg.train.device == "cpu" and torch.cuda.is_available():
        cfg.train.device = "cuda"
    model, vocab, lexicon, train_loader, val_loader, pool_loader = build_everything(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                              weight_decay=cfg.train.weight_decay)
    pool_iter = itertools.cycle(pool_loader) if pool_loader is not None else None

    print(f"[train] lexicon={len(lexicon)} ({lexicon.source}) "
          f"vocab={vocab.size} device={cfg.train.device} "
          f"dorsal_pool={'on' if pool_iter else 'off'}")
    history = []
    for ep in range(cfg.train.epochs):
        tr = run_epoch(model, train_loader, cfg, optim, pool_iter=pool_iter)
        with torch.no_grad():
            va = run_epoch(model, val_loader, cfg, optim=None)
        history.append({"epoch": ep + 1,
                        "train_total": tr["total"], "train_rep": tr["rep"],
                        "train_wm": tr["wm"], "val_rep": va["rep"],
                        "val_wm": va["wm"]})
        print(f"[ep {ep+1:2d}/{cfg.train.epochs}] "
              f"train total={tr['total']:.3f} rep={tr['rep']:.3f} "
              f"align={tr['align']:.3f} wm={tr['wm']:.3f} | "
              f"val rep={va['rep']:.3f} wm={va['wm']:.3f}")
    if out_dir is not None:
        import os
        plot_loss_history(history, os.path.join(out_dir, "training_loss.png"))
    return model, vocab, lexicon, history, optim


if __name__ == "__main__":
    from config import default_config
    build_and_train(default_config(), out_dir="outputs")
