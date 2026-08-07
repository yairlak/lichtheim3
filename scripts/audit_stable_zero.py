"""Stable-zero checkpoint-selection audit for the full-lexicon cohort.

**No training, no inference, no checkpoint is loaded.**  This reads the frozen
aggregate evaluation tables shipped in the cohort bundle and re-derives every
zero-error streak from the raw per-epoch error counts.

Criterion under audit, as specified:

    the selected checkpoint is the FIRST checkpoint of a streak of X
    consecutive evaluated zero-error checkpoints; training can stop only once
    the Xth zero has been observed.

Two consequences the audit reports separately:

  * `selected_epoch`   the first checkpoint of the first qualifying streak
                       (what you would *use*)
  * `stop_epoch`       the Xth checkpoint of that streak
                       (the earliest point at which you could *know* it qualified,
                       and therefore the earliest epoch training could stop)

Missing evaluations are never inferred: the evaluated epoch grid is reported as
found, and a gap would be flagged rather than interpolated.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BUNDLE = ("archives/fulllexicon_93a577f/extracted/"
          "fulllexicon_final_bundle_93a577f/aggregate_results")
ALL_CKPT = "all_checkpoints.tsv"
SELECTED = "selected_checkpoints.tsv"
SEED_SUMMARY = "seed_summary.tsv"

OUT = ("reports/behavioral_wfe_fulllexicon_93a577f/yair_corrections/"
       "stable_zero_audit")

X_VALUES = (2, 3, 5)
# The train-AR error count the cohort selection used: FULL-route autoregressive
# errors over the 29,571-word training lexicon.
ERROR_COLUMN = "n_errors_full"


def streaks_of_zeros(epochs: List[int], errors: List[int]) -> List[dict]:
    """Every maximal run of consecutive evaluated zero-error checkpoints."""
    out, start = [], None
    for i, e in enumerate(errors):
        if e == 0 and start is None:
            start = i
        if (e != 0 or i == len(errors) - 1) and start is not None:
            end = i if e == 0 else i - 1
            out.append({"start_index": start, "end_index": end,
                        "start_epoch": epochs[start], "end_epoch": epochs[end],
                        "length": end - start + 1,
                        "epochs": ",".join(str(epochs[k])
                                           for k in range(start, end + 1))})
            start = None
    return out


def audit(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    traj_rows, streak_rows, verdict_rows = [], [], []
    for seed in sorted(df["seed"].unique()):
        d = df[df["seed"] == seed].sort_values("epoch")
        epochs = [int(e) for e in d["epoch"]]
        errors = [int(v) for v in d[ERROR_COLUMN]]
        steps = sorted({epochs[i + 1] - epochs[i] for i in range(len(epochs) - 1)})
        grid_regular = len(steps) == 1
        for e, v in zip(epochs, errors):
            traj_rows.append({"seed": int(seed), "epoch": e,
                              "train_ar_errors_full": v,
                              "is_zero": int(v == 0)})
        st = streaks_of_zeros(epochs, errors)
        for k, sdict in enumerate(st, 1):
            streak_rows.append({"seed": int(seed), "streak_id": k,
                                "first_checkpoint_of_streak": sdict["start_epoch"],
                                "last_checkpoint_of_streak": sdict["end_epoch"],
                                "length": sdict["length"],
                                "epochs": sdict["epochs"]})
        longest = max((s["length"] for s in st), default=0)
        for X in X_VALUES:
            qual = [s for s in st if s["length"] >= X]
            first = qual[0] if qual else None
            verdict_rows.append({
                "seed": int(seed), "X": X,
                "criterion_met": bool(first is not None),
                "n_qualifying_streaks": len(qual),
                "longest_zero_streak": longest,
                "selected_epoch": first["start_epoch"] if first else None,
                "stop_epoch_earliest_knowable":
                    (epochs[first["start_index"] + X - 1] if first else None),
                "qualifying_streak_epochs": first["epochs"] if first else "",
                "n_evaluated_checkpoints": len(epochs),
                "evaluated_epoch_range": f"{epochs[0]}-{epochs[-1]}",
                "evaluation_step": steps[0] if grid_regular else ";".join(
                    map(str, steps)),
                "evaluation_grid_regular": grid_regular,
                "missing_evaluations_inferred": False,
            })
    return {"trajectory": pd.DataFrame(traj_rows),
            "streaks": pd.DataFrame(streak_rows),
            "verdicts": pd.DataFrame(verdict_rows)}


def main() -> int:
    src = os.path.join(ROOT, BUNDLE, ALL_CKPT)
    df = pd.read_csv(src, sep="\t")
    res = audit(df)
    out_dir = os.path.join(ROOT, OUT)
    os.makedirs(out_dir, exist_ok=True)
    for name, t in res.items():
        t.to_csv(os.path.join(out_dir, f"stable_zero_{name}.tsv"), sep="\t",
                 index=False)

    # cross-check against the cohort's own selection record
    sel = pd.read_csv(os.path.join(ROOT, BUNDLE, SELECTED), sep="\t")
    v2 = res["verdicts"][res["verdicts"]["X"] == 2].set_index("seed")
    cross = []
    for _, r in sel.iterrows():
        s = int(r["seed"])
        cross.append({
            "seed": s,
            "cohort_selected_epoch": int(r["selected_epoch"]),
            "cohort_selection_reason": r["selection_reason"],
            "cohort_selected_errors": int(r["n_errors_full"]),
            "audit_X2_selected_epoch": v2.loc[s, "selected_epoch"],
            "audit_X2_criterion_met": bool(v2.loc[s, "criterion_met"]),
            "agrees_with_cohort_selection":
                (bool(v2.loc[s, "criterion_met"])
                 and int(v2.loc[s, "selected_epoch"]) == int(r["selected_epoch"])),
        })
    pd.DataFrame(cross).to_csv(
        os.path.join(out_dir, "stable_zero_cross_check.tsv"), sep="\t",
        index=False)

    summary = {
        "source_table": os.path.join(BUNDLE, ALL_CKPT),
        "error_column": ERROR_COLUMN,
        "error_column_meaning":
            "FULL-route autoregressive error count over the 29,571-word "
            "training lexicon at that evaluated epoch",
        "training_performed": False, "inference_performed": False,
        "checkpoint_loaded": False,
        "missing_evaluations_inferred": False,
        "criterion": ("selected checkpoint = first checkpoint of a streak of X "
                      "consecutive evaluated zero-error checkpoints; training "
                      "can stop only when the Xth zero is observed"),
        "X_values_audited": list(X_VALUES),
        "verdict_by_X": {
            str(X): {
                "seeds_meeting": sorted(int(s) for s in
                                        res["verdicts"][(res["verdicts"]["X"] == X)
                                        & res["verdicts"]["criterion_met"]]["seed"]),
                "seeds_failing": sorted(int(s) for s in
                                        res["verdicts"][(res["verdicts"]["X"] == X)
                                        & ~res["verdicts"]["criterion_met"]]["seed"]),
            } for X in X_VALUES},
    }
    with open(os.path.join(out_dir, "stable_zero_audit.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(res["verdicts"].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
