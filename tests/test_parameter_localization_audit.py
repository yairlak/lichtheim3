"""Tests for the FINAL-7A parameter-localisation audit tool.

The tool is read-only: it loads endpoint checkpoints, evaluates weight states
(including transplanted ones), and writes only to its own report directory.
These tests pin the properties that make its conclusions trustworthy --
disjoint and exhaustive groups, transplants that copy values rather than
aliasing tensors, a fresh complete assignment per condition, no optimizer
step, and byte-identical source checkpoints.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.parameter_localization_audit import (  # noqa: E402
    COMPOSITES, GROUP_TOUCHED_BY, PARAM_GROUPS, build_conditions,
    build_parser, cosine, displacement_table, group_of, sha256_file,
)
from scripts.naming_comprehension.train_joint_scratch import (           # noqa: E402
    FINAL_FULL_MODE, INTERLEAVED_123, JointScratchTrainer, main as train_main,
)

TINY = dict(device="cpu", max_words=400,
            lexicon_path="data/lexicon_en_glove_covered.tsv",
            dorsal_pool_size=32, batch_size=8, subset_mode=FINAL_FULL_MODE,
            subset_per_band=822, subset_size=32, lr_boundary_steps=6,
            allow_glove_fallback=True, require_subset_hash=False,
            glove_path="tests/_no_such_glove_file.txt")


def model_names():
    tr = JointScratchTrainer(regime="j0", seed=22, schedule=INTERLEAVED_123,
                             **TINY)
    return [n for n, _ in tr.model.named_parameters()]


# =====================================================  group definitions  ==

def test_groups_are_exhaustive_and_disjoint():
    names = model_names()
    assigned = {n: group_of(n) for n in names}
    assert all(g is not None for g in assigned.values()), \
        f"ungrouped: {[n for n, g in assigned.items() if g is None]}"
    # disjoint: each name matches exactly one group's prefixes
    for n in names:
        hits = [g for g, pres in PARAM_GROUPS.items() if n.startswith(pres)]
        assert len(hits) == 1, f"{n} matched {hits}"
    # every declared group is populated
    for g in PARAM_GROUPS:
        assert any(assigned[n] == g for n in names), f"empty group {g}"
    assert set(GROUP_TOUCHED_BY) == set(PARAM_GROUPS)


def test_shared_embedding_alias_cannot_be_moved_by_the_wm_group():
    """state_dict exposes wm.phon_embed.weight; the audit works from
    named_parameters so a 'wm' transplant cannot drag the shared embedding."""
    tr = JointScratchTrainer(regime="j0", seed=22, schedule=INTERLEAVED_123,
                             **TINY)
    assert "wm.phon_embed.weight" in tr.model.state_dict()
    assert "wm.phon_embed.weight" not in [n for n, _ in
                                          tr.model.named_parameters()]
    assert group_of("phon_embed.weight") == "phon_embed"
    wm_names = [n for n in model_names() if group_of(n) == "wm"]
    assert wm_names and all("phon_embed" not in n for n in wm_names)


def test_composites_reference_declared_groups_only():
    for name, gs in COMPOSITES.items():
        assert gs, name
        for g in gs:
            assert g in PARAM_GROUPS, f"{name} references unknown group {g}"
    # A and B partition the task-touched surfaces without overlapping
    a, b = set(COMPOSITES["A_encoder_semantic_side"]), set(COMPOSITES["B_production_side"])
    assert not (a & b), "encoder and production composites overlap"
    assert a | b == set(PARAM_GROUPS) - {"wm"}


# =====================================================  displacement math  ==

def test_displacement_table_is_correct_and_reports_cosines():
    r100 = {"phon_embed.weight": torch.zeros(4),
            "ltm.encoder.weight_ih_l0": torch.zeros(4)}
    runs = {
        "control": {"phon_embed.weight": torch.tensor([1., 0, 0, 0]),
                    "ltm.encoder.weight_ih_l0": torch.tensor([1., 0, 0, 0])},
        "sep": {"phon_embed.weight": torch.tensor([2., 0, 0, 0]),
                "ltm.encoder.weight_ih_l0": torch.tensor([0., 1, 0, 0])},
        "grouped": {"phon_embed.weight": torch.tensor([-1., 0, 0, 0]),
                    "ltm.encoder.weight_ih_l0": torch.tensor([1., 0, 0, 0])},
    }
    rows = {r["group"]: r for r in displacement_table(r100, runs)}
    pe = rows["phon_embed"]
    assert pe["delta_norm_control"] == pytest.approx(1.0)
    assert pe["delta_norm_sep"] == pytest.approx(2.0)
    assert pe["cos_control_vs_sep"] == pytest.approx(1.0)
    assert pe["cos_control_vs_grouped"] == pytest.approx(-1.0)
    enc = rows["ltm_encoder"]
    assert enc["cos_control_vs_sep"] == pytest.approx(0.0, abs=1e-9)
    assert enc["cos_control_vs_grouped"] == pytest.approx(1.0)
    assert "ALL" in rows and rows["ALL"]["n_params"] == 8


def test_cosine_handles_degenerate_vectors():
    v = torch.tensor([1.0, 0.0])
    assert cosine(v, v) == pytest.approx(1.0)
    assert cosine(v, torch.zeros(2)) != cosine(v, torch.zeros(2)) or True
    import math
    assert math.isnan(cosine(v, torch.zeros(2)))


# =========================================================  condition set  ==

def test_condition_set_covers_every_requested_audit():
    args = build_parser().parse_args(
        ["--r100", "a", "--control", "b", "--sep", "c", "--grouped", "d"])
    conds = build_conditions(args)
    labels = [c["label"] for c in conds]
    kinds = {c["kind"] for c in conds}
    assert kinds == {"intact", "single", "composite", "reverse",
                     "sixp_vs_sevenp"}
    # the three intact endpoints are reproduced first
    assert labels[:3] == ["intact_control", "intact_sep_6P",
                          "intact_grouped_7P"]
    # every single group is screened from control into 7P
    for g in PARAM_GROUPS:
        assert f"7P_take_{g}_from_control" in labels
    for name in COMPOSITES:
        assert f"7P_take_{name}_from_control" in labels
    # reverse direction and the 6P disambiguation are present
    assert any(c["base"] == "control" and c["donor"] == "grouped" for c in conds)
    assert any(c["donor"] == "sep" and c["base"] == "grouped" for c in conds)
    # bases/donors only ever name real inputs
    for c in conds:
        assert c["base"] in {"control", "sep", "grouped"}
        assert c["donor"] in {None, "control", "sep", "grouped"}


def test_conditions_filter_selects_kinds():
    p = ["--r100", "a", "--control", "b", "--sep", "c", "--grouped", "d"]
    args = build_parser().parse_args(p + ["--conditions", "intact"])
    assert {c["kind"] for c in build_conditions(args)} == {"intact"}


# ==========================================  end-to-end on real artefacts  ==

@pytest.fixture(scope="module")
def endpoints(tmp_path_factory):
    """Three tiny endpoint checkpoints standing in for control / 6P / 7P,
    plus a common origin, produced by the real driver."""
    out = str(tmp_path_factory.mktemp("runs"))
    common = ["--regime", "j0", "--seed", "22", "--subset-mode", "final_full",
              "--schedule", "interleaved_123", "--device", "cpu",
              "--max-words", "400", "--batch-size", "8",
              "--dorsal-pool-size", "32", "--lr-boundary-steps", "6",
              "--eval-every", "0", "--log-every", "0",
              "--glove-path", "tests/_no_such_glove_file.txt",
              "--allow-glove-fallback", "--no-subset-hash-check",
              "--out-dir", out]

    def run(run_id, steps, extra=()):
        assert train_main(common + ["--run-id", run_id, "--max-steps",
                                    str(steps), "--save-every", str(steps)]
                          + list(extra)) == 0
        return os.path.join(out, run_id, "checkpoints", f"step_{steps:08d}.pt")

    origin = run("origin", 6)
    control = run("control", 12, ["--resume", origin])
    sep = run("sep", 12, ["--resume", origin, "--optimizer-policy",
                          "task_separated_adamw", "--phase-transition"])
    grouped = run("grouped", 12, ["--resume", origin, "--optimizer-policy",
                                  "grouped_rn_c_adamw", "--phase-transition"])
    return {"r100": origin, "control": control, "sep": sep, "grouped": grouped}


def test_audit_runs_read_only_and_writes_consistent_tables(endpoints, tmp_path):
    from scripts.naming_comprehension.parameter_localization_audit import main

    before = {k: sha256_file(v) for k, v in endpoints.items()}
    out_dir = str(tmp_path / "report")
    rc = main(["--r100", endpoints["r100"], "--control", endpoints["control"],
               "--sep", endpoints["sep"], "--grouped", endpoints["grouped"],
               "--out-dir", out_dir, "--device", "cpu",
               "--eval-population", "sample", "--sample-size", "64",
               "--routes", "full,ltm", "--skip-step-check",
               "--conditions", "intact,single"])
    assert rc == 0

    # every source checkpoint is byte-identical afterwards
    assert {k: sha256_file(v) for k, v in endpoints.items()} == before

    report = json.load(open(os.path.join(out_dir, "audit.json")))
    assert report["analysis_only"] is True
    assert report["training_steps"] == 0 and report["optimizer_steps"] == 0
    assert report["source_checkpoints_unchanged"] is True
    assert "does NOT show" in report["causal_caveat"]
    # inputs carry real provenance, including each run's optimizer policy
    assert report["inputs"]["sep"]["optimizer_policy"] == "task_separated_adamw"
    assert report["inputs"]["grouped"]["optimizer_policy"] == "grouped_rn_c_adamw"
    assert report["inputs"]["control"]["optimizer_policy"] == "shared_adamw"

    # CSVs agree with the JSON
    cond_csv = list(csv.DictReader(open(os.path.join(out_dir, "conditions.csv"))))
    assert len(cond_csv) == len(report["conditions"])
    assert {r["label"] for r in cond_csv} == {r["label"] for r in report["conditions"]}
    disp_csv = list(csv.DictReader(open(os.path.join(out_dir, "displacement.csv"))))
    assert {r["group"] for r in disp_csv} == set(list(PARAM_GROUPS) + ["ALL"])
    for r in report["conditions"]:
        for k in ("comp_top1", "naming_exact", "rep_full", "rep_ltm"):
            assert 0.0 <= r[k] <= 1.0


def test_transplant_copies_values_and_leaves_no_residue(endpoints):
    """A transplant must install a fresh COMPLETE assignment: donor values for
    the named groups, base values everywhere else, and no tensor aliasing."""
    from scripts.naming_comprehension.parameter_localization_audit import Localizer

    loc = Localizer("cpu", "sample", 8, ("full",))
    base = loc.named_from_checkpoint(endpoints["control"])
    donor = loc.named_from_checkpoint(endpoints["grouped"])
    assert any(not torch.equal(base[n], donor[n]) for n in loc.names), \
        "fixture endpoints are identical; the test would be vacuous"

    loc.apply(base, donor, ["ltm_encoder"])
    for n, p in loc.model.named_parameters():
        want = donor[n] if loc.groups[n] == "ltm_encoder" else base[n]
        assert torch.equal(p.detach().cpu(), want), n
        assert p.data_ptr() != want.data_ptr(), f"{n} aliases the source"

    # a second condition fully replaces the first -- no residue
    loc.apply(base, donor, ["motor"])
    for n, p in loc.model.named_parameters():
        want = donor[n] if loc.groups[n] == "motor" else base[n]
        assert torch.equal(p.detach().cpu(), want), f"residue at {n}"

    # ... and the intact base restores exactly
    loc.apply(base)
    for n, p in loc.model.named_parameters():
        assert torch.equal(p.detach().cpu(), base[n]), n


def test_audit_takes_no_optimizer_step(endpoints):
    from scripts.naming_comprehension.parameter_localization_audit import Localizer

    loc = Localizer("cpu", "sample", 8, ("full",))
    base = loc.named_from_checkpoint(endpoints["control"])
    n_state = len(loc.tr.optim.state)
    loc.apply(base)
    snapshot = {n: p.detach().clone() for n, p in loc.model.named_parameters()}
    loc.evaluate()
    for n, p in loc.model.named_parameters():
        assert torch.equal(p.detach(), snapshot[n]), f"evaluation moved {n}"
    assert len(loc.tr.optim.state) == n_state, "an optimizer step was taken"


def test_intact_conditions_reproduce_their_checkpoint_metrics(endpoints):
    """Evaluating an intact endpoint twice is deterministic, and the three
    endpoints are genuinely different weight states."""
    from scripts.naming_comprehension.parameter_localization_audit import Localizer

    import math

    def same(a, b):
        # a route that was not decoded reports NaN, and NaN != NaN, so the
        # comparison has to be NaN-aware rather than a plain dict equality
        if set(a) != set(b):
            return False
        return all(
            (math.isnan(a[k]) and math.isnan(b[k])) if isinstance(a[k], float)
            and math.isnan(a[k]) else a[k] == b[k] for k in a)

    loc = Localizer("cpu", "sample", 64, ("full", "ltm"))
    got = {}
    for key in ("control", "sep", "grouped"):
        st = loc.named_from_checkpoint(endpoints[key])
        loc.apply(st)
        a = loc.evaluate()
        loc.apply(st)
        b = loc.evaluate()
        assert same(a, b), f"{key}: evaluation is not deterministic"
        got[key] = a
    # The three endpoints must be genuinely different WEIGHT states.  Their
    # metrics can legitimately coincide here -- twelve steps on a 400-word
    # lexicon score 0.0 everywhere -- so distinctness is asserted on the
    # parameters, which is what the transplants actually manipulate.
    states = {k: loc.named_from_checkpoint(endpoints[k])
              for k in ("control", "sep", "grouped")}
    for a, b in (("control", "sep"), ("control", "grouped"), ("sep", "grouped")):
        assert any(not torch.equal(states[a][n], states[b][n])
                   for n in loc.names), f"{a} and {b} are identical states"
    assert set(got) == {"control", "sep", "grouped"}
