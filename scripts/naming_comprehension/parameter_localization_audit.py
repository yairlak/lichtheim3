#!/usr/bin/env python3
"""FINAL-7A: read-only parameter-localisation audit at the matched R130 state.

Three runs branch from one R100 checkpoint and differ ONLY in optimizer-state
topology (shared R+N+C, separated R|N|C, grouped RN|C).  At R130 they show a
striking dissociation, so this audit asks WHERE in parameter space the
comprehension gain and the LTM-repetition loss live.

It does three things, all read-only:

  A. displacement of every parameter group from the common R100 state, with
     pairwise cosines between the runs' displacement directions;
  B. weight transplants -- take one group (or composite) from a donor endpoint
     into a base endpoint and re-evaluate, which localises which parameters
     the endpoint phenotype DEPENDS on;
  C. the reverse direction, which separates necessity from sufficiency.

CAUSAL CAVEAT, carried into every output: transplantation is a diagnostic
intervention on endpoint parameter states.  It localises endpoint dependence
and is consistent with (or against) a mechanism; it does NOT show that
training a group differently would reproduce the trajectory.

Nothing is ever written back to a run directory, no optimizer is constructed
for stepping, and no backward pass is taken.  Source checkpoints are opened
read-only and fingerprinted before and after.

Usage:
    python scripts/naming_comprehension/parameter_localization_audit.py \
        --r100    .../final3p_.../checkpoints/step_00277800.pt \
        --control .../final3p_.../checkpoints/step_00361140.pt \
        --sep     .../final6p_.../checkpoints/step_00361140.pt \
        --grouped .../final7p_.../checkpoints/step_00361140.pt \
        --device cuda --out-dir reports/final7_parameter_localization_r130
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from scripts.naming_comprehension.train_joint_scratch import (           # noqa: E402
    FINAL_FULL_MODE, INTERLEAVED_123, NAMING_MAX_STEPS, JointScratchTrainer,
    deterministic_sample,
)
from scripts.naming_comprehension.train_tasks import (                   # noqa: E402
    evaluate_comprehension_subset, evaluate_naming, repetition_snapshot,
)

EXPECTED_STEP = 361_140

# ---------------------------------------------------------------------------
# Parameter groups, by the model's CANONICAL named_parameters() names.
# state_dict() also exposes the shared embedding as wm.phon_embed.weight and
# ltm.phon_embed.weight; working from named_parameters() keeps every group
# disjoint and stops a "wm" transplant from silently moving the shared
# embedding through an alias.
# ---------------------------------------------------------------------------
PARAM_GROUPS: "OrderedDict[str, tuple]" = OrderedDict([
    ("phon_embed",       ("phon_embed.",)),                 # shared, C and N
    ("ltm_encoder",      ("ltm.encoder.",)),                # C only (+R)
    ("to_semantic",      ("ltm.to_semantic.",)),            # C only (+R)
    ("sem_to_h0",        ("ltm.sem_to_h0.",)),              # N only (+R)
    ("ltm_decoder",      ("ltm.decoder.",)),                # N only (+R)
    ("dec_to_premotor",  ("ltm.dec_to_premotor.",)),        # N only (+R)
    ("motor",            ("motor.",)),                      # N, R, pool
    ("wm",               ("wm.",)),                         # R and pool only
])

# Which tasks' gradients reach each group (established in the FINAL-4 audit).
GROUP_TOUCHED_BY = {
    "phon_embed": "R, N, C", "ltm_encoder": "R, C", "to_semantic": "R, C",
    "sem_to_h0": "R, N", "ltm_decoder": "R, N", "dec_to_premotor": "R, N",
    "motor": "R, N, pool", "wm": "R, pool",
}

COMPOSITES: "OrderedDict[str, List[str]]" = OrderedDict([
    # A: everything a comprehension step can touch
    ("A_encoder_semantic_side", ["phon_embed", "ltm_encoder", "to_semantic"]),
    # B: the production pathway naming and ventral repetition use, which a
    #    comprehension step never touches (shared embedding excluded)
    ("B_production_side", ["sem_to_h0", "ltm_decoder", "dec_to_premotor",
                           "motor"]),
    # C: the lower phonological encoder alone
    ("C_lower_encoder", ["ltm_encoder"]),
    # D: the semantic projection alone
    ("D_to_semantic", ["to_semantic"]),
    # E: production core without the shared readout
    ("E_production_core", ["sem_to_h0", "ltm_decoder", "dec_to_premotor"]),
    # extra: the whole ventral route, and the dorsal buffer
    ("F_ventral_all", ["ltm_encoder", "to_semantic", "sem_to_h0",
                       "ltm_decoder", "dec_to_premotor"]),
    ("G_dorsal_wm", ["wm"]),
])


def group_of(name: str) -> Optional[str]:
    for g, prefixes in PARAM_GROUPS.items():
        if name.startswith(prefixes):
            return g
    return None


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flat(named: Dict[str, torch.Tensor], names: Sequence[str]) -> torch.Tensor:
    parts = [named[n].reshape(-1).double() for n in names]
    return torch.cat(parts) if parts else torch.zeros(1, dtype=torch.double)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = float(a.norm()), float(b.norm())
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float((a @ b) / (na * nb))


class Localizer:
    """Owns one model instance and evaluates weight states on it.

    A "condition" is a full parameter assignment built from a base state and
    (optionally) a donor's groups.  Every condition is applied by copying
    VALUES into the model's parameters, so no condition can alias another and
    none can leak into the next.
    """

    def __init__(self, device: str, eval_population: str, sample_size: int,
                 routes: Sequence[str]) -> None:
        self.device = device
        self.routes = tuple(routes)
        self.tr = JointScratchTrainer(
            regime="j0", seed=22, device=device, max_words=30000,
            lexicon_path="data/lexicon_en_glove_covered.tsv",
            dorsal_pool_size=4000, batch_size=64, subset_mode=FINAL_FULL_MODE,
            subset_per_band=822, subset_size=64, lr_boundary_steps=46_300,
            allow_glove_fallback=False, require_subset_hash=True,
            glove_path="data/glove.6B.300d.txt", schedule=INTERLEAVED_123)
        self.model = self.tr.model
        self.names = [n for n, _ in self.model.named_parameters()]
        self.groups = {n: group_of(n) for n in self.names}
        ungrouped = [n for n, g in self.groups.items() if g is None]
        if ungrouped:
            raise RuntimeError(f"parameters not covered by any group: {ungrouped}")

        all_idx = list(range(len(self.tr.entries)))
        comp_idx = list(self.tr.comp_idx)
        if eval_population == "sample":
            all_idx = deterministic_sample(all_idx, sample_size, 4_000_000)
            comp_idx = deterministic_sample(comp_idx, sample_size, 4_000_001)
        self.rep_idx, self.nam_idx, self.comp_idx = all_idx, all_idx, comp_idx
        self.eval_population = eval_population

    # ------------------------------------------------------------- states
    def named_from_checkpoint(self, path: str) -> Dict[str, torch.Tensor]:
        """The canonical named_parameters view of a checkpoint's weights.

        Read-only: the file is loaded, the tensors are cloned to CPU, and the
        checkpoint object is discarded.
        """
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"]
        missing = [n for n in self.names if n not in sd]
        if missing:
            raise RuntimeError(f"{path}: checkpoint lacks parameters {missing}")
        return {n: sd[n].detach().clone() for n in self.names}

    def apply(self, base: Dict[str, torch.Tensor],
              donor: Optional[Dict[str, torch.Tensor]] = None,
              groups: Sequence[str] = ()) -> None:
        """Install a fresh, complete parameter assignment on the model."""
        gset = set(groups)
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                src = donor if (donor is not None and self.groups[n] in gset) \
                    else base
                p.copy_(src[n].to(p.device, p.dtype))

    # -------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        was = self.model.training
        self.model.eval()
        rep = repetition_snapshot(self.model, self.tr.vocab, self.tr.entries,
                                  self.rep_idx, self.tr.bank_raw, self.device,
                                  routes=self.routes,
                                  include_teacher_forced=False)
        exact = rep["primary_readout"]["exact_match"]
        comp = evaluate_comprehension_subset(
            self.model, self.tr.vocab, self.tr.entries, self.tr.bank_raw,
            self.comp_idx, self.device)
        nam = evaluate_naming(self.model, self.tr.vocab, self.tr.entries,
                              self.tr.bank_raw, self.nam_idx, self.device,
                              NAMING_MAX_STEPS)
        self.model.train(was)
        n_rep = len(self.rep_idx)
        out = {
            "rep_full": exact.get("full", float("nan")),
            "rep_ltm": exact.get("ltm", float("nan")),
            "rep_wm": exact.get("wm", float("nan")),
            "rep_full_errors": (1.0 - exact.get("full", float("nan"))) * n_rep,
            "rep_ltm_errors": (1.0 - exact.get("ltm", float("nan"))) * n_rep,
            "comp_top1": comp["top1"], "comp_top5": comp["top5"],
            "comp_rank_median": comp["target_rank_median"],
            "naming_exact": nam["exact_match"],
            "naming_wer": nam["whole_word_error_rate"],
            "n_rep_items": n_rep, "n_comp_items": len(self.comp_idx),
        }
        return out


# ===========================================================  audit A  ======

def displacement_table(r100: Dict[str, torch.Tensor],
                       runs: Dict[str, Dict[str, torch.Tensor]]
                       ) -> List[dict]:
    rows: List[dict] = []
    group_names = list(PARAM_GROUPS) + ["ALL"]
    for g in group_names:
        names = ([n for n in r100 if group_of(n) == g] if g != "ALL"
                 else list(r100))
        base = flat(r100, names)
        deltas = {k: flat(v, names) - base for k, v in runs.items()}
        row = {"group": g, "touched_by": GROUP_TOUCHED_BY.get(g, "R, N, C"),
               "n_params": int(base.numel()),
               "theta_r100_norm": float(base.norm())}
        for k, d in deltas.items():
            row[f"delta_norm_{k}"] = float(d.norm())
            row[f"delta_rel_{k}"] = float(d.norm()) / max(float(base.norm()), 1e-12)
        keys = list(deltas)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                row[f"cos_{a}_vs_{b}"] = cosine(deltas[a], deltas[b])
        rows.append(row)
    return rows


# =========================================================  conditions  =====

def build_conditions(args) -> List[dict]:
    """(label, base, donor, groups) for every evaluation this audit runs."""
    conds: List[dict] = [
        {"label": "intact_control", "base": "control", "donor": None,
         "groups": [], "kind": "intact"},
        {"label": "intact_sep_6P", "base": "sep", "donor": None,
         "groups": [], "kind": "intact"},
        {"label": "intact_grouped_7P", "base": "grouped", "donor": None,
         "groups": [], "kind": "intact"},
    ]
    # B: single groups and composites, CONTROL -> FINAL-7P
    for g in PARAM_GROUPS:
        conds.append({"label": f"7P_take_{g}_from_control", "base": "grouped",
                      "donor": "control", "groups": [g], "kind": "single"})
    for name, gs in COMPOSITES.items():
        conds.append({"label": f"7P_take_{name}_from_control", "base": "grouped",
                      "donor": "control", "groups": gs, "kind": "composite"})
    # C: reverse direction, FINAL-7P -> CONTROL, on the informative surfaces
    for name in ("A_encoder_semantic_side", "B_production_side",
                 "C_lower_encoder", "D_to_semantic"):
        conds.append({"label": f"control_take_{name}_from_7P", "base": "control",
                      "donor": "grouped", "groups": COMPOSITES[name],
                      "kind": "reverse"})
    # D: 6P -> 7P, which differ mainly through RN sharing
    for name in ("B_production_side", "E_production_core",
                 "A_encoder_semantic_side"):
        conds.append({"label": f"7P_take_{name}_from_6P", "base": "grouped",
                      "donor": "sep", "groups": COMPOSITES[name],
                      "kind": "sixp_vs_sevenp"})
    if args.conditions:
        wanted = {c.strip() for c in args.conditions.split(",")}
        conds = [c for c in conds if c["kind"] in wanted or c["label"] in wanted]
    return conds


# ==============================================================  main  ======

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--r100", required=True)
    p.add_argument("--control", required=True)
    p.add_argument("--sep", required=True, help="FINAL-6P R130 (R|N|C)")
    p.add_argument("--grouped", required=True, help="FINAL-7P R130 (RN|C)")
    p.add_argument("--out-dir", default="reports/final7_parameter_localization_r130")
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval-population", choices=("full", "sample"),
                   default="full")
    p.add_argument("--sample-size", type=int, default=4096,
                   help="only with --eval-population sample (fast screen)")
    p.add_argument("--routes", default="full,ltm,wm",
                   help="repetition routes to decode; drop 'wm' to save time")
    p.add_argument("--conditions", default=None,
                   help="comma-separated kinds (intact,single,composite,"
                        "reverse,sixp_vs_sevenp) or explicit labels")
    p.add_argument("--skip-step-check", action="store_true",
                   help="allow endpoints that are not step 361,140 (testing)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    paths = {"r100": args.r100, "control": args.control, "sep": args.sep,
             "grouped": args.grouped}
    before = {k: sha256_file(v) for k, v in paths.items()}

    # --- provenance of every input, verified not assumed --------------------
    meta = {}
    for key, path in paths.items():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        meta[key] = {
            "path": os.path.abspath(path),
            "sha256": before[key],
            "global_step": int(ck["global_step"]),
            "cursors": {k: int(v) for k, v in ck["cursors"].items()},
            "optimizer_policy": ck.get("optimizer_policy", "shared_adamw"),
            "optimizer_bank_layout": ck.get("optimizer_bank_layout"),
            "schedule": ck.get("schedule"),
            "lr_policy": ck.get("lr_policy"),
            "git_commit": (ck.get("git") or {}).get("commit"),
        }
        want = 277_800 if key == "r100" else EXPECTED_STEP
        if not args.skip_step_check and meta[key]["global_step"] != want:
            raise SystemExit(
                f"{key}: expected step {want}, got {meta[key]['global_step']}")
        del ck
    print("[audit] inputs verified:")
    for k, m in meta.items():
        print(f"  {k:8s} step={m['global_step']:>7} policy={m['optimizer_policy']:22s} "
              f"sha={m['sha256'][:12]}")

    loc = Localizer(args.device, args.eval_population, args.sample_size,
                    [r.strip() for r in args.routes.split(",") if r.strip()])
    print(f"[audit] evaluation population: {args.eval_population} "
          f"(rep/naming {len(loc.rep_idx)}, comprehension {len(loc.comp_idx)}), "
          f"routes {loc.routes}")

    states = {k: loc.named_from_checkpoint(v) for k, v in paths.items()}

    # --- AUDIT A ------------------------------------------------------------
    disp = displacement_table(states["r100"],
                              {k: states[k] for k in ("control", "sep", "grouped")})
    print("\n== A. displacement from the common R100 state ==")
    print(f"  {'group':22s}{'touched':12s}{'|d|ctrl':>10}{'|d|6P':>10}"
          f"{'|d|7P':>10}{'cos c~6':>9}{'cos c~7':>9}{'cos 6~7':>9}")
    for r in disp:
        print(f"  {r['group']:22s}{r['touched_by']:12s}"
              f"{r['delta_norm_control']:>10.4f}{r['delta_norm_sep']:>10.4f}"
              f"{r['delta_norm_grouped']:>10.4f}"
              f"{r['cos_control_vs_sep']:>+9.3f}{r['cos_control_vs_grouped']:>+9.3f}"
              f"{r['cos_sep_vs_grouped']:>+9.3f}")

    # --- AUDITS B / C / D ---------------------------------------------------
    conds = build_conditions(args)
    print(f"\n== B/C/D. {len(conds)} weight states to evaluate ==")
    results: List[dict] = []
    for i, c in enumerate(conds, 1):
        loc.apply(states[c["base"]],
                  states[c["donor"]] if c["donor"] else None, c["groups"])
        t = time.time()
        m = loc.evaluate()
        row = {"label": c["label"], "kind": c["kind"], "base": c["base"],
               "donor": c["donor"] or "", "groups": "+".join(c["groups"]),
               "param_names": ";".join(n for n in loc.names
                                       if loc.groups[n] in set(c["groups"])),
               "seconds": round(time.time() - t, 1)}
        row.update(m)
        results.append(row)
        print(f"  [{i:>2}/{len(conds)}] {c['label']:44s} "
              f"C1={m['comp_top1']:.5f} N={m['naming_exact']:.5f} "
              f"LTM={m['rep_ltm']:.5f} FULL={m['rep_full']:.5f} "
              f"({row['seconds']}s)", flush=True)

    # --- outputs ------------------------------------------------------------
    after = {k: sha256_file(v) for k, v in paths.items()}
    unchanged = all(before[k] == after[k] for k in paths)
    report = {
        "analysis_only": True, "training_steps": 0, "optimizer_steps": 0,
        "inputs": meta,
        "source_checkpoints_unchanged": unchanged,
        "evaluation": {
            "population": args.eval_population,
            "n_rep_naming_items": len(loc.rep_idx),
            "n_comprehension_targets": len(loc.comp_idx),
            "routes": list(loc.routes),
            "definitions": ("identical to the training driver: canonical "
                            "forced-length AR repetition, free greedy AR "
                            "naming from true GloVe, full-bank cosine "
                            "retrieval over the canonical C population"),
        },
        "param_groups": {g: list(p) for g, p in PARAM_GROUPS.items()},
        "group_touched_by": GROUP_TOUCHED_BY,
        "composites": {k: v for k, v in COMPOSITES.items()},
        "causal_caveat": (
            "Weight transplantation is a diagnostic intervention on ENDPOINT "
            "parameter states. It localises endpoint dependence and is "
            "consistent with or against a mechanism; it does NOT show that "
            "training a group differently would reproduce the trajectory."),
        "displacement": disp,
        "conditions": results,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    if not unchanged:                                        # pragma: no cover
        raise SystemExit("FATAL: a source checkpoint changed during the audit")

    json.dump(report, open(os.path.join(args.out_dir, "audit.json"), "w"),
              indent=2, default=str)
    for name, rows in (("displacement.csv", disp), ("conditions.csv", results)):
        if not rows:
            continue
        cols: List[str] = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with open(os.path.join(args.out_dir, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"\n[audit] source checkpoints unchanged: {unchanged}")
    print(f"[audit] wrote {args.out_dir}/audit.json, displacement.csv, "
          f"conditions.csv  ({report['runtime_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
