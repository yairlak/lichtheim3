"""CAP-1: from-scratch full-lexicon naming-only VENTRAL-WIDTH diagnostic.

Question.  The semantic->phonology mapping is learnable at small scale
(subset 3,288 reached 100%; ~10k blocks reached 97.2-98.4% when trained
separately, all warm-started at width 128), but the SAME words trained as the
full 29,571-word naming lexicon reached only 9.77% exact after 3,000
exposures/item (phase2h, warm start, hidden 128, LR 1e-4).  Hidden size 128
was chosen historically to reach ceiling on REPETITION.  Is ventral capacity
the bottleneck at full lexical scale?

Design.  One knob: VENTRAL DECODER WIDTH, i.e. `cfg.ltm.dec_hidden`, which
sets sem_to_h0's output, the production GRU's hidden state, and
dec_to_premotor's input.  The dorsal WM route stays at its canonical 128 (it
already reaches ceiling and is untouched by the naming loss anyway), and so
does the ventral ENCODER (the naming objective never calls it: naming is
RAW GloVe -> decode).  Everything else is the canonical recipe: same seed,
same counter-addressed naming stream (identical item order at every width),
same optimizer, LR, batch, clipping, losses, and the committed free-greedy
evaluation (BOS -> greedy -> EOS/cap, target length never consulted).

Training is FROM SCRATCH (unlike phase2, which warm-started from the
repetition checkpoint and froze phon_embed + motor.proj).  Gradients flow
only through the naming path -- phon_embed, ltm.sem_to_h0, ltm.decoder,
ltm.dec_to_premotor, motor.proj -- because no other loss is computed; AdamW
never touches a parameter whose grad is None, so the dorsal route and the
ventral encoder keep their init values, and that is asserted at every
checkpoint.

Distinguishing capacity failure from autoregressive compounding: alongside
whole-word exact match under free greedy decoding, each evaluation records
full-population naming CE, TEACHER-FORCED token accuracy, and GREEDY
positional token accuracy.  High TF accuracy with poor exact match points at
sequence compounding; high CE points at fitting/capacity.

This is a LOCAL diagnostic (Mac); `--benchmark N` times N steps and exits.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.dual_route import DualRouteModel                              # noqa: E402
from scripts.naming_comprehension.train_joint_scratch import (            # noqa: E402
    CANONICAL_HIDDEN, NAMING_MAX_STEPS, CounterStream, build_batch,
    canonical_config, capture_rng_states, derive_stream_seeds,
    restore_rng_states,
)
from scripts.naming_comprehension.train_tasks import (                    # noqa: E402
    evaluate_naming, naming_objective,
)
from data.lexicon import build_lexicon                                    # noqa: E402
from data.phonemes import build_vocab                                     # noqa: E402
from utils.provenance import git_state, sha256_file                       # noqa: E402

FORMAT = "lichtheim3.naming_capacity.v1"
CANONICAL_WIDTHS = (128, 256, 512)
LR = 1e-3                     # from-scratch stage-1 LR; constant, NOT tuned
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
BATCH_SIZE = 64
EVAL_EXPOSURES = (0, 100, 300, 500, 750, 1000, 1500)

# Parameters the naming loss can reach.  Everything else must stay at init.
NAMING_PATH_PREFIXES = ("phon_embed.", "ltm.sem_to_h0.", "ltm.decoder.",
                        "ltm.dec_to_premotor.", "motor.")

METRIC_COLUMNS = ["step", "exposures", "train_ce_running", "full_ce",
                  "tf_token_acc", "greedy_token_acc", "exact_match",
                  "whole_word_error_rate", "mean_edit", "eos_rate",
                  "mean_pred_len", "mean_target_len", "elapsed_s"]


def param_census(model: DualRouteModel) -> Dict[str, int]:
    groups = {"total": 0, "wm": 0, "ltm_encoder": 0, "ltm_sem_to_h0": 0,
              "ltm_decoder": 0, "ltm_dec_to_premotor": 0, "phon_embed": 0,
              "motor": 0, "gate": 0, "naming_path": 0}
    for name, p in model.named_parameters():
        n = p.numel()
        groups["total"] += n
        if name.startswith("wm."):
            groups["wm"] += n
        elif name.startswith(("ltm.encoder", "ltm.to_semantic", "ltm.phon_embed")):
            groups["ltm_encoder"] += n
        elif name.startswith("ltm.sem_to_h0."):
            groups["ltm_sem_to_h0"] += n
        elif name.startswith("ltm.decoder."):
            groups["ltm_decoder"] += n
        elif name.startswith("ltm.dec_to_premotor."):
            groups["ltm_dec_to_premotor"] += n
        elif name.startswith("phon_embed."):
            groups["phon_embed"] += n
        elif name.startswith("motor."):
            groups["motor"] += n
        elif name.startswith("gate."):
            groups["gate"] += n
        if name.startswith(NAMING_PATH_PREFIXES):
            groups["naming_path"] += n
    # the shared embedding is aliased under ltm.phon_embed in named_parameters
    return groups


def frozen_fingerprint(model: DualRouteModel) -> Dict[str, torch.Tensor]:
    """Everything the naming loss must NOT move (dorsal + ventral encoder)."""
    return {n: p.detach().clone().cpu() for n, p in model.named_parameters()
            if not n.startswith(NAMING_PATH_PREFIXES)}


def assert_frozen_untouched(model: DualRouteModel,
                            before: Dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    moved = [n for n, ref in before.items()
             if not torch.equal(params[n].detach().cpu(), ref)]
    if moved:
        raise RuntimeError(
            f"non-naming parameters changed in a naming-only run: {moved[:8]}")


class NamingCapacityTrainer:
    def __init__(self, *, width: int, seed: int, device: str,
                 max_words: int = 30000, batch_size: int = BATCH_SIZE,
                 lr: float = LR,
                 lexicon_path: str = "data/lexicon_en_glove_covered.tsv",
                 glove_path: Optional[str] = "data/glove.6B.300d.txt",
                 allow_glove_fallback: bool = False) -> None:
        self.width = int(width)
        self.seed = int(seed)
        self.lr = float(lr)
        self.lr_transitions: List[dict] = []
        self.device = device
        torch.manual_seed(self.seed)

        cfg = canonical_config(self.seed, device, max_words=max_words,
                               lexicon_path=lexicon_path,
                               dorsal_pool_size=4000, batch_size=batch_size,
                               glove_path=glove_path)
        # THE single knob.  WM and the ventral encoder stay canonical.
        cfg.ltm.dec_hidden = self.width
        assert cfg.wm.hidden == CANONICAL_HIDDEN
        assert cfg.ltm.enc_hidden == CANONICAL_HIDDEN
        self.cfg = cfg

        self.vocab = build_vocab()
        self.lexicon = build_lexicon(cfg.data, self.vocab)
        self.entries = list(self.lexicon.entries)
        stats = self.lexicon.load_stats
        self.glove_fallback = int(getattr(stats, "n_glove_fallback", 0))
        if self.glove_fallback and not allow_glove_fallback:
            raise RuntimeError(
                f"{self.glove_fallback} entries lack real GloVe; the capacity "
                f"sweep requires real vectors (pass --allow-glove-fallback "
                f"only in tests)")
        self.bank_raw = torch.stack(
            [torch.tensor(e.semantic) for e in self.entries]).float()

        self.model = DualRouteModel(cfg, self.vocab).to(device)
        self.model.set_semantic_bank(self.bank_raw.to(device))
        self.frozen_ref = frozen_fingerprint(self.model)

        # naming population: the FULL lexicon, no homophone restriction
        self.naming_idx = list(range(len(self.entries)))
        seeds = derive_stream_seeds(self.seed)
        self.stream = CounterStream("naming", self.naming_idx, batch_size,
                                    seeds["naming"])
        self.per_epoch = self.stream.per_epoch

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                       weight_decay=WEIGHT_DECAY)
        self.global_step = 0
        self.running_ce: Optional[float] = None

    # ------------------------------------------------------------------ train
    def train_step(self) -> float:
        self.model.train()
        batch = build_batch(self.entries, self.bank_raw, self.vocab,
                            self.stream.indices(self.global_step), self.device)
        out = naming_objective(self.model, batch, self.vocab.pad_id)
        loss = out["total"]
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
        self.optim.step()
        self.global_step += 1
        ce = float(loss.detach())
        self.running_ce = (ce if self.running_ce is None
                           else 0.99 * self.running_ce + 0.01 * ce)
        return ce

    # ------------------------------------------------------------------- eval
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        # 1) free greedy AR over the full population (committed definition)
        nam = evaluate_naming(self.model, self.vocab, self.entries,
                              self.bank_raw, self.naming_idx, self.device,
                              NAMING_MAX_STEPS, return_per_item=True)
        per_item = nam.pop("_per_item", [])
        # greedy positional token accuracy from the same decode
        tok_ok = tok_all = 0
        pred_len_sum = tgt_len_sum = 0
        for row in per_item:
            gold = [self.vocab.itos[p]
                    for p in self.entries[row["bank_index"]].phonemes]
            pred = row["pred"].split()
            tok_all += len(gold)
            tok_ok += sum(int(k < len(pred) and pred[k] == g)
                          for k, g in enumerate(gold))
            pred_len_sum += row["pred_len"]
            tgt_len_sum += len(gold)
        # 2) teacher-forced CE + token accuracy over the full population
        ce_sum = ce_tok = tf_ok = tf_all = 0
        for lo in range(0, len(self.naming_idx), 512):
            idx = self.naming_idx[lo:lo + 512]
            batch = build_batch(self.entries, self.bank_raw, self.vocab, idx,
                                self.device)
            out = naming_objective(self.model, batch, self.vocab.pad_id)
            tgt = batch["dec_tgt"]
            mask = tgt != self.vocab.pad_id
            n_tok = int(mask.sum())
            ce_sum += float(out["total"]) * n_tok
            ce_tok += n_tok
            pred = out["logits"].argmax(-1)
            tf_ok += int((pred[mask] == tgt[mask]).sum())
            tf_all += n_tok
        return {"full_ce": ce_sum / max(ce_tok, 1),
                "tf_token_acc": tf_ok / max(tf_all, 1),
                "greedy_token_acc": tok_ok / max(tok_all, 1),
                "exact_match": nam["exact_match"],
                "whole_word_error_rate": nam["whole_word_error_rate"],
                "mean_edit": nam["mean_edit"],
                "eos_rate": nam["eos_emission_rate"],
                "mean_pred_len": pred_len_sum / max(len(per_item), 1),
                "mean_target_len": tgt_len_sum / max(len(per_item), 1)}

    # ------------------------------------------------------------- checkpoint
    def state_dict(self) -> dict:
        return {"format": FORMAT, "width": self.width, "seed": self.seed,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                "running_ce": self.running_ce,
                "per_epoch": self.per_epoch,
                "lr": self.lr, "lr_transitions": list(self.lr_transitions),
                "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP,
                "batch_size": self.stream.batch_size,
                "rng_states": capture_rng_states(),
                "git": git_state(ROOT)}

    def load_state_dict(self, ckpt: dict, *,
                        allow_lr_transition: bool = False) -> None:
        if ckpt.get("format") != FORMAT:
            raise RuntimeError(f"unexpected format {ckpt.get('format')!r}")
        if int(ckpt["width"]) != self.width or int(ckpt["seed"]) != self.seed:
            raise RuntimeError(
                f"checkpoint is width {ckpt['width']} seed {ckpt['seed']}; "
                f"this trainer is width {self.width} seed {self.seed}")
        if int(ckpt["per_epoch"]) != self.per_epoch:
            raise RuntimeError("population size changed between runs")
        self.model.load_state_dict(ckpt["model_state_dict"])
        # AdamW moments restored exactly; load_state_dict also overwrites the
        # param-group lr with the checkpoint's, so the requested lr is applied
        # explicitly afterwards -- moments are NEVER reset by a transition.
        self.optim.load_state_dict(ckpt["optimizer_state_dict"])
        ck_lr = float(ckpt.get("lr", LR))
        self.lr_transitions = list(ckpt.get("lr_transitions", []))
        if self.lr != ck_lr:
            if not allow_lr_transition:
                raise RuntimeError(
                    f"LR TRANSITION REFUSED: checkpoint lr {ck_lr} != requested "
                    f"{self.lr}. Pass --lr-transition to declare it.")
            self.lr_transitions.append({
                "from_lr": ck_lr, "to_lr": self.lr,
                "at_step": int(ckpt["global_step"]),
                "at_exposures": round(int(ckpt["global_step"])
                                      / int(ckpt["per_epoch"]), 4),
                "optimizer_moments": "preserved"})
        for g in self.optim.param_groups:
            g["lr"] = self.lr
        self.global_step = int(ckpt["global_step"])
        self.running_ce = ckpt.get("running_ce")
        restore_rng_states(ckpt["rng_states"])
        # the frozen reference must describe the RESUMED weights
        self.frozen_ref = frozen_fingerprint(self.model)


# ===========================================================================

def append_row(path: str, row: Dict[str, object]) -> None:
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write("\t".join(METRIC_COLUMNS) + "\n")
        f.write("\t".join(str(row.get(c, "")) for c in METRIC_COLUMNS) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--width", type=int, required=True,
                    help="ventral decoder width (cfg.ltm.dec_hidden)")
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-words", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--eval-exposures", default=",".join(map(str, EVAL_EXPOSURES)))
    ap.add_argument("--max-exposures", type=int, default=EVAL_EXPOSURES[-1])
    ap.add_argument("--resume", default=None)
    ap.add_argument("--lr", type=float, default=LR,
                    help="optimizer LR; differs from a resumed checkpoint's "
                         "only with --lr-transition (moments preserved)")
    ap.add_argument("--lr-transition", action="store_true",
                    help="declare an intentional LR change on resume")
    ap.add_argument("--no-stop-at-ceiling", action="store_true",
                    help="keep training after exact_match reaches 1.0")
    ap.add_argument("--benchmark", type=int, default=0,
                    help="time N optimizer steps and exit (no eval, no save)")
    ap.add_argument("--allow-glove-fallback", action="store_true")
    ap.add_argument("--lexicon-path", default="data/lexicon_en_glove_covered.tsv")
    ap.add_argument("--glove-path", default="data/glove.6B.300d.txt")
    ap.add_argument("--log-every", type=int, default=463)
    args = ap.parse_args(argv)

    tr = NamingCapacityTrainer(
        width=args.width, seed=args.seed, device=args.device,
        max_words=args.max_words, batch_size=args.batch_size, lr=args.lr,
        lexicon_path=args.lexicon_path, glove_path=args.glove_path,
        allow_glove_fallback=args.allow_glove_fallback)

    census = param_census(tr.model)
    print(f"[capacity] width={args.width} device={args.device} "
          f"pop={len(tr.naming_idx)} per_epoch={tr.per_epoch}")
    print(f"[capacity] params: {json.dumps(census)}")

    if args.benchmark:
        for _ in range(5):                                   # warm-up
            tr.train_step()
        if args.device == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        for _ in range(args.benchmark):
            tr.train_step()
        if args.device == "mps":
            torch.mps.synchronize()
        dt = (time.time() - t0) / args.benchmark
        print(f"[capacity] benchmark width={args.width}: {dt:.6f} s/step "
              f"({1.0/dt:.1f} steps/s); 1 exposure = {tr.per_epoch} steps = "
              f"{dt*tr.per_epoch:.1f} s; 1500 exposures ~= "
              f"{dt*tr.per_epoch*1500/3600:.2f} h")
        return 0

    run_id = args.run_id or f"cap1_w{args.width}_seed{args.seed}"
    run_dir = os.path.join(args.out_dir, run_id)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics = os.path.join(run_dir, "metrics.tsv")

    parent_sha = None
    if args.resume:
        import hashlib
        parent_sha = hashlib.sha256(
            open(args.resume, "rb").read()).hexdigest()
        tr.load_state_dict(torch.load(args.resume, map_location=args.device,
                                      weights_only=False),
                           allow_lr_transition=args.lr_transition)
        print(f"[capacity] resumed step {tr.global_step} "
              f"({tr.global_step/tr.per_epoch:.1f} exp) from {args.resume}")
        print(f"[capacity] parent sha256 {parent_sha}")
        if tr.lr_transitions:
            print(f"[capacity] lr transitions: {tr.lr_transitions}")

    cfgdump = {"format": FORMAT, "width": args.width, "seed": args.seed,
               "device": args.device, "lr": args.lr,
               "lr_schedule": "constant (single declared transition allowed)",
               "lr_transitions": list(tr.lr_transitions),
               "parent_checkpoint": args.resume,
               "parent_checkpoint_sha256": parent_sha,
               "resumed_at_step": (tr.global_step if args.resume else 0),
               "resumed_at_exposures": (round(tr.global_step / tr.per_epoch, 4)
                                        if args.resume else 0.0),
               "optimizer_moments": ("preserved from parent" if args.resume
                                     else "fresh"),
               "lexicon_file_sha256": sha256_file(
                   os.path.join(ROOT, args.lexicon_path)),
               "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
               "batch_size": args.batch_size,
               "population": len(tr.naming_idx), "per_epoch": tr.per_epoch,
               "wm_hidden": tr.cfg.wm.hidden,
               "ltm_enc_hidden": tr.cfg.ltm.enc_hidden,
               "ltm_dec_hidden": tr.cfg.ltm.dec_hidden,
               "from_scratch": True, "trained_scope": list(NAMING_PATH_PREFIXES),
               "eval": "free greedy AR, BOS->EOS/cap, no target length",
               "params": census, "git": git_state(ROOT),
               "eval_exposures": args.eval_exposures,
               "max_exposures": args.max_exposures}
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        json.dump(cfgdump, open(cfg_path, "w"), indent=1)

    eval_steps = sorted({int(e) * tr.per_epoch
                         for e in args.eval_exposures.split(",")})
    max_steps = args.max_exposures * tr.per_epoch
    t0 = time.time()

    def do_eval() -> Dict[str, object]:
        row = {"step": tr.global_step,
               "exposures": round(tr.global_step / tr.per_epoch, 4),
               "train_ce_running": ("" if tr.running_ce is None
                                    else round(tr.running_ce, 6))}
        row.update({k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in tr.evaluate().items()})
        row["elapsed_s"] = round(time.time() - t0, 1)
        append_row(metrics, row)
        assert_frozen_untouched(tr.model, tr.frozen_ref)
        torch.save(tr.state_dict(),
                   os.path.join(ckpt_dir, f"step_{tr.global_step:08d}.pt"))
        print(f"  [EVAL @ {row['exposures']} exp] exact={row['exact_match']} "
              f"ce={row['full_ce']} tf_tok={row['tf_token_acc']} "
              f"greedy_tok={row['greedy_token_acc']}", flush=True)
        return row

    ceiling = False
    if tr.global_step in eval_steps:
        row0 = do_eval()          # step-0 baseline, or resume continuity point
        ceiling = (row0["exact_match"] == 1.0
                   and not args.no_stop_at_ceiling)
    while tr.global_step < max_steps and not ceiling:
        ce = tr.train_step()
        if args.log_every and tr.global_step % args.log_every == 0:
            print(f"[step {tr.global_step}/{max_steps}] "
                  f"exp={tr.global_step/tr.per_epoch:.1f} ce={ce:.4f} "
                  f"run={tr.running_ce:.4f} | {time.time()-t0:.0f}s",
                  flush=True)
        if tr.global_step in eval_steps:
            row = do_eval()
            if row["exact_match"] == 1.0 and not args.no_stop_at_ceiling:
                print(f"[capacity] CEILING REACHED: exact 1.0 over the full "
                      f"population at {row['exposures']} exposures; "
                      f"checkpoint saved; stopping this branch.", flush=True)
                ceiling = True
    if tr.global_step not in eval_steps and not ceiling:
        do_eval()
    print(f"[capacity] done: width={args.width} step={tr.global_step} "
          f"({tr.global_step/tr.per_epoch:.1f} exposures) -> {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
