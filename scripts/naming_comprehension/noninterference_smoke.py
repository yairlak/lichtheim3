"""V6 NON-INTERFERENCE GATE.

Does inserting a full-lexicon detector evaluation alter SUBSEQUENT training?

Two INDEPENDENT processes build the trainer from the SAME checkpoint:

  A : load -> K training steps
  B : load -> full-lexicon evaluation -> K training steps

Then A and B are compared bitwise on model parameters, optimizer moments,
explicit Generator states, the global RNG state, cursors, rep_epoch and
global_step.  Two fresh builds are used deliberately: resetting one trainer
in-process would NOT reset the dorsal pool's itertools.cycle cursor -- exactly
the state the historical resume failed to checkpoint -- and would therefore
fake the comparison.

B additionally snapshots immediately before and after the evaluation, so a
perturbation is localised to the evaluation itself.

PASS requires bitwise equality everywhere.  Anything else is BLOCKED.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GLOVE = os.path.join(ROOT, "data", "glove.6B.300d.txt")


def _gen_states(tr) -> dict:
    """Every torch.Generator reachable on the trainer / its streams."""
    out = {}
    def scan(obj, prefix):
        for name in dir(obj):
            if name.startswith("__"):
                continue
            try:
                v = getattr(obj, name)
            except Exception:
                continue
            if isinstance(v, torch.Generator):
                out[f"{prefix}.{name}"] = v.get_state().clone()
    scan(tr, "trainer")
    streams = getattr(tr, "streams", None)
    if isinstance(streams, dict):
        for k, s in streams.items():
            scan(s, f"stream[{k}]")
    pool = getattr(tr, "pool", None)
    if pool is not None:
        scan(pool, "pool")
    return out


def snapshot(tr) -> dict:
    model = {k: v.detach().cpu().clone() for k, v in tr.model.state_dict().items()}
    opt = tr.optimizer.state_dict() if hasattr(tr, "optimizer") else {}
    def clean(o):
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().clone()
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    return {
        "model": model,
        "optimizer": clean(opt),
        "generators": _gen_states(tr),
        "global_rng": torch.get_rng_state().clone(),
        "global_step": int(tr.global_step),
        "rep_epoch": int(getattr(tr, "rep_epoch", -1)),
        "cursors": dict(getattr(tr, "cursors", {}) or {}),
    }


def diff(a: dict, b: dict, label_a: str, label_b: str) -> list:
    """Bitwise comparison; returns the list of mismatching keys."""
    bad = []
    for k in sorted(set(a["model"]) | set(b["model"])):
        ta, tb = a["model"].get(k), b["model"].get(k)
        if ta is None or tb is None or not torch.equal(ta, tb):
            bad.append(f"model.{k}")
    def cmp(x, y, path):
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
                    and x.shape == y.shape and torch.equal(x, y)):
                bad.append(path)
            return
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                cmp(x.get(k), y.get(k), f"{path}.{k}")
            return
        if isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                bad.append(f"{path}.len"); return
            for i, (u, v) in enumerate(zip(x, y)):
                cmp(u, v, f"{path}[{i}]")
            return
        if x != y:
            bad.append(path)
    cmp(a["optimizer"], b["optimizer"], "optimizer")
    for k in sorted(set(a["generators"]) | set(b["generators"])):
        ga, gb = a["generators"].get(k), b["generators"].get(k)
        if ga is None or gb is None or not torch.equal(ga, gb):
            bad.append(f"generator.{k}")
    if not torch.equal(a["global_rng"], b["global_rng"]):
        bad.append("global_rng")
    for k in ("global_step", "rep_epoch", "cursors"):
        if a[k] != b[k]:
            bad.append(k)
    print(f"[smoke] compare {label_a} vs {label_b}: "
          f"{'BITWISE EQUAL' if not bad else str(len(bad)) + ' MISMATCHES'}")
    for k in bad[:20]:
        print(f"          MISMATCH {k}")
    return bad


def cmd_arm(a) -> int:
    torch.set_num_threads(a.nt)
    from scripts.naming_comprehension.base123_error_audit import build
    tr, ck = build(a.ckpt, "cpu", glove_path=GLOVE)
    print(f"[smoke] arm {a.arm}: loaded step={tr.global_step} "
          f"rep_epoch={tr.rep_epoch} cursors={tr.cursors}", flush=True)
    out = {"arm": a.arm, "ckpt": a.ckpt, "steps": a.steps}
    if a.arm == "B":
        out["pre_eval"] = snapshot(tr)
        row = tr.evaluate(with_probe=True, with_full_lexicon=True)
        out["eval_row"] = {k: v for k, v in row.items()
                           if isinstance(v, (int, float, str))}
        out["post_eval"] = snapshot(tr)
        print(f"[smoke] arm B: full detector eval done "
              f"full_rep_errors={row.get('full_rep_errors')} "
              f"full_rep_freear_errors={row.get('full_rep_freear_errors')} "
              f"full_naming_exact={row.get('full_naming_exact')}", flush=True)
    for i in range(a.steps):
        tr.train_step()
    out["final"] = snapshot(tr)
    print(f"[smoke] arm {a.arm}: after {a.steps} steps step={tr.global_step}",
          flush=True)
    torch.save(out, a.out)
    print(f"[smoke] arm {a.arm}: wrote {a.out}", flush=True)
    return 0


def cmd_compare(a) -> int:
    A = torch.load(a.a, map_location="cpu", weights_only=False)
    B = torch.load(a.b, map_location="cpu", weights_only=False)
    print("=" * 70)
    print("NON-INTERFERENCE GATE")
    print("=" * 70)
    bad_eval = diff(B["pre_eval"], B["post_eval"],
                    "B before eval", "B after eval")
    bad_final = diff(A["final"], B["final"],
                     "A (K steps)", "B (eval + K steps)")
    ok = not bad_eval and not bad_final
    print()
    print(f"  evaluation itself perturbs state : {'NO' if not bad_eval else 'YES'}")
    print(f"  subsequent training state equal  : {'YES' if not bad_final else 'NO'}")
    print(f"  A final step = {A['final']['global_step']}   "
          f"B final step = {B['final']['global_step']}")
    print()
    print(f"NON_INTERFERENCE_GATE={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("arm")
    r.add_argument("--arm", choices=("A", "B"), required=True)
    r.add_argument("--ckpt", required=True)
    r.add_argument("--steps", type=int, default=12)
    r.add_argument("--out", required=True)
    r.add_argument("--nt", type=int, default=2)
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    a = ap.parse_args(argv)
    return cmd_arm(a) if a.cmd == "arm" else cmd_compare(a)


if __name__ == "__main__":
    raise SystemExit(main())
