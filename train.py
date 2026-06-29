"""Training loop and a `build_and_train` convenience entry point."""
from __future__ import annotations

from typing import Tuple

import torch

from config import Config
from data.phonemes import build_vocab, Vocab
from data.lexicon import build_lexicon, Lexicon
from data.dataset import make_loader
from models.dual_route import DualRouteModel
from losses import total_loss
from utils.seed import set_seed


def build_everything(cfg: Config):
    set_seed(cfg.train.seed)
    vocab = build_vocab()
    lexicon = build_lexicon(cfg.data, vocab)
    density = lexicon.neighborhood_density()
    train_entries, val_entries = lexicon.split(cfg.data.val_fraction, cfg.data.seed)

    train_loader = make_loader(
        train_entries, vocab, density, cfg.train.batch_size,
        frequency_weighted=True, freq_temp=cfg.data.freq_temp, shuffle=True)
    val_loader = make_loader(
        val_entries, vocab, density, cfg.train.batch_size,
        frequency_weighted=False, shuffle=False)

    model = DualRouteModel(cfg, vocab).to(cfg.train.device)
    # frozen lexical bank = GloVe vectors of the *training* lexicon
    bank = torch.stack([torch.tensor(e.semantic) for e in train_entries]).float()
    model.set_semantic_bank(bank.to(cfg.train.device))
    return model, vocab, lexicon, train_loader, val_loader


def run_epoch(model, loader, cfg: Config, optim=None) -> dict:
    train_mode = optim is not None
    model.train(train_mode)
    pad_id = model.vocab.pad_id
    agg = {}
    n = 0
    for batch in loader:
        batch = {k: (v.to(cfg.train.device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        out = model(batch["enc_in"], batch["enc_mask"], batch["dec_in"])
        losses = total_loss(out, batch, cfg.loss, pad_id,
                            usage_prior=cfg.gating.usage_prior)
        if train_mode:
            optim.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optim.step()
        for k, v in losses.items():
            agg[k] = agg.get(k, 0.0) + float(v) * len(batch["words"])
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
                    ) -> Tuple[DualRouteModel, Vocab, Lexicon, list]:
    if cfg.train.device == "cpu" and torch.cuda.is_available():
        cfg.train.device = "cuda"
    model, vocab, lexicon, train_loader, val_loader = build_everything(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                              weight_decay=cfg.train.weight_decay)

    print(f"[train] lexicon={len(lexicon)} ({lexicon.source}) "
          f"vocab={vocab.size} device={cfg.train.device}")
    history = []
    for ep in range(cfg.train.epochs):
        tr = run_epoch(model, train_loader, cfg, optim)
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
    return model, vocab, lexicon, history


if __name__ == "__main__":
    from config import default_config
    build_and_train(default_config(), out_dir="outputs")
