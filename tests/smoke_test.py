"""Fast end-to-end smoke test (requires torch).

    python -m tests.smoke_test

Builds a tiny synthetic model, runs one training epoch, checks every output
shape, and executes all four evaluations on a handful of items. Finishes in
seconds on CPU. Use this to confirm your environment before a full `run_all.py`.
"""
from __future__ import annotations

import tempfile

import torch

from config import default_config
from train import build_everything, run_epoch
from evaluate import primacy_recency, neighborhood, sonority, dissociation


def main():
    cfg = default_config()
    cfg.data.use_real = False        # self-contained
    cfg.data.max_words = 250
    cfg.train.epochs = 1
    cfg.train.batch_size = 32
    cfg.train.device = "cpu"

    model, vocab, lexicon, train_loader, val_loader = build_everything(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

    # one forward/backward
    stats = run_epoch(model, train_loader, cfg, optim)
    assert torch.isfinite(torch.tensor(stats["total"])), stats

    # shape checks on a single batch
    batch = next(iter(val_loader))
    out = model(batch["enc_in"], batch["enc_mask"], batch["dec_in"])
    B, S = batch["dec_in"].shape
    assert out["logits"].shape == (B, S, vocab.size)
    assert out["wm_logits"].shape == (B, S, vocab.size)
    assert out["ltm_logits"].shape == (B, S, vocab.size)
    assert out["s_hat"].shape == (B, cfg.data.semantic_dim)
    assert out["gate"].shape == (B, S, 1)
    assert out["gate"].min() >= 0 and out["gate"].max() <= 1

    with tempfile.TemporaryDirectory() as d:
        primacy_recency.run(model, vocab, cfg, d, lengths=(5, 7), n_words=24,
                            n_trials=2)
        neighborhood.run(model, vocab, lexicon, cfg, d, max_per_group=24)
        sonority.run(model, vocab, cfg, d, n_words=40, length=5, n_trials=3)
        dissociation.run(model, vocab, lexicon, cfg, d, n_trials=2)
    print("  [ok] forward, losses, and evaluations\n")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
