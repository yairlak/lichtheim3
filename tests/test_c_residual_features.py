"""Tests for the read-only residual-C characterisation.

The analysis must (a) use only metadata the repo actually has, (b) recover a
known lexical bias when one is present, (c) separate the four-way persistent
core from churn-only items, and (d) never treat pooled cross-seed counts as
an iid sample.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.c_residual_features import (          # noqa: E402
    ARPABET_VOWELS, CONTINUOUS, dist, main, smd,
)

AUDIT = "scripts/cluster/jeanzay/rescue123_u1400_error_audit.slurm"
GLOVE = "data/glove.6B.300d.txt"
SEEDS = [19, 20, 21, 22]
COLS = ["task", "target_bank_index", "target_word", "target_rank", "in_top5",
        "pred_word", "margin_target_minus_top1", "pred_target_glove_cos",
        "target_cos", "is_homophone_of_target", "identical_glove_vectors",
        "mathematically_unavoidable", "relation"]

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, GLOVE)),
    reason="needs the real GloVe; fallback is never permitted for this analysis")


def script(path=AUDIT):
    return open(os.path.join(ROOT, path), encoding="utf-8").read()


def _canonical():
    from config import default_config
    from data.lexicon import build_lexicon
    from data.phonemes import build_vocab
    from scripts.naming_comprehension.train_tasks import (
        canonical_phonology_indices)
    cfg = default_config()
    cfg.data.use_real = True
    cfg.data.lexicon_path = "data/lexicon_en_glove_covered.tsv"
    cfg.data.glove_path = GLOVE
    cfg.data.max_words = 30000
    cfg.data.split_mode = "full_lexicon"
    cfg.data.val_fraction = 0.0
    v = build_vocab()
    lex = build_lexicon(cfg.data, v)
    e = list(lex.entries)
    return e, canonical_phonology_indices(e)


def _fixture(root, entries, chosen, core, relation_core="morphological"):
    for s in SEEDS:
        d = os.path.join(root, f"final_rep_rescue123_h512_s{s}",
                         "error_audit_u1400")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "comp_errors.tsv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, delimiter="\t")
            w.writeheader()
            for i in core + chosen[s]:
                w.writerow({
                    "task": "comprehension", "target_bank_index": i,
                    "target_word": entries[i].word, "target_rank": 2,
                    "in_top5": 1, "pred_word": "x",
                    "margin_target_minus_top1": -0.006,
                    "pred_target_glove_cos": 0.8, "target_cos": 0.7,
                    "is_homophone_of_target": 0, "identical_glove_vectors": 0,
                    "mathematically_unavoidable": 0,
                    "relation": relation_core if i in core
                    else "semantic_neighbour"})


def _run(root, out):
    return main(["--runs-root", root, "--out-dir", out,
                 "--run-template", "final_rep_rescue123_h512_s{seed}",
                 "--u", "1400", "--label", "t", "--glove-path", GLOVE,
                 "--no-plots"])


def _rows(out, name):
    with open(os.path.join(out, name), encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_only_real_metadata_is_used():
    """LexEntry has word/phonemes/freq/rank -- no syllable or morphology
    field -- so the proxy must be labelled and nothing invented."""
    import inspect
    from scripts.naming_comprehension import c_residual_features as m
    src = inspect.getsource(m)
    assert "syllable_proxy" in src and '"syllables"' not in src
    assert len(ARPABET_VOWELS) == 15
    assert {"IY", "AH", "ER", "OY"} <= ARPABET_VOWELS
    assert "P" not in ARPABET_VOWELS and "NG" not in ARPABET_VOWELS
    assert set(CONTINUOUS) == {"orth_length", "phon_length", "syllable_proxy",
                               "log10_rank", "freq"}


def test_recovers_a_planted_length_and_rarity_bias(tmp_path):
    entries, cidx = _canonical()
    ranked = sorted(cidx, key=lambda i: (-len(entries[i].phonemes),
                                         -entries[i].rank))
    core = ranked[:12]
    chosen = {s: ranked[12 + k * 60: 12 + k * 60 + 80]
              for k, s in enumerate(SEEDS)}
    root, out = str(tmp_path / "runs"), str(tmp_path / "out")
    _fixture(root, entries, chosen, core)
    assert _run(root, out) == 0
    summ = {(r["set"], r["feature"]): r
            for r in _rows(out, "residual_vs_population_summary.tsv")}
    for feat in ("phon_length", "orth_length", "log10_rank"):
        d = float(summ[("t:pooled", feat)]["standardized_mean_diff"])
        assert d > 0.5, f"{feat} bias not recovered (SMD {d})"
        assert float(summ[("t:pooled", feat)]["resid_median"]) >= \
            float(summ[("t:pooled", feat)]["pop_median"])
    cons = {r["feature"]: r for r in _rows(out, "seed_consistency.tsv")}
    for feat in ("phon_length", "orth_length", "log10_rank"):
        assert cons[feat]["same_sign_all_seeds"] == "1"
        assert int(cons[feat]["n_seeds"]) == 4


def test_reports_no_bias_when_residuals_are_a_random_sample(tmp_path):
    """The complement: a null fixture must NOT produce a large effect."""
    import random
    entries, cidx = _canonical()
    rng = random.Random(11)
    core = rng.sample(cidx, 10)
    chosen = {s: rng.sample(cidx, 90) for s in SEEDS}
    root, out = str(tmp_path / "runs"), str(tmp_path / "out")
    _fixture(root, entries, chosen, core)
    assert _run(root, out) == 0
    summ = {(r["set"], r["feature"]): r
            for r in _rows(out, "residual_vs_population_summary.tsv")}
    for feat in ("phon_length", "orth_length", "log10_rank"):
        d = abs(float(summ[("t:pooled", feat)]["standardized_mean_diff"]))
        assert d < 0.4, f"{feat} spurious effect {d} on a random sample"


def test_core_and_churn_are_separated(tmp_path):
    entries, cidx = _canonical()
    ranked = sorted(cidx, key=lambda i: -len(entries[i].phonemes))
    core = ranked[:12]
    chosen = {s: ranked[100 + k * 50: 100 + k * 50 + 40]
              for k, s in enumerate(SEEDS)}
    root, out = str(tmp_path / "runs"), str(tmp_path / "out")
    _fixture(root, entries, chosen, core)
    assert _run(root, out) == 0
    meta = json.load(open(os.path.join(out, "residual_meta.json")))
    assert meta["n_four_way_core"] == 12
    assert meta["n_pooled"] == 4 * (12 + 40)
    assert meta["n_unique"] < meta["n_pooled"]
    assert meta["n_churn_only"] > 0
    assert "NOT iid" in meta["note"]
    assert "ARPABET vowel count" in meta["syllable_proxy"]
    corerows = _rows(out, "persistent_core.tsv")
    assert len(corerows) == 12
    assert all(r["n_seeds_wrong"] == "4" for r in corerows)
    sets = {r["set"] for r in _rows(out, "residual_vs_population_summary.tsv")}
    for want in ("t:pooled", "t:unique", "t:four_way_core", "t:churn_only"):
        assert want in sets
    assert {f"t:seed{s}" for s in SEEDS} <= sets


def test_categorical_enrichment_is_relative_to_the_population(tmp_path):
    entries, cidx = _canonical()
    ranked = sorted(cidx, key=lambda i: -len(entries[i].phonemes))
    core, chosen = ranked[:12], {s: ranked[12:92] for s in SEEDS}
    root, out = str(tmp_path / "runs"), str(tmp_path / "out")
    _fixture(root, entries, chosen, core)
    assert _run(root, out) == 0
    cat = _rows(out, "residual_categorical.tsv")
    rel = [r for r in cat if r["variable"] == "relation"
           and r["set"] == "t:pooled"]
    assert rel and sum(int(r["count"]) for r in rel) == 4 * 92
    homo = [r for r in cat if r["variable"] == "homophone_group_size"
            and r["set"] == "t:pooled"]
    assert homo and all(r["population_proportion"] for r in homo)
    assert any(r["enrichment_vs_population"] for r in homo)


def test_effect_size_helpers():
    assert smd([1, 1, 1], [1, 1, 1]) is None          # zero population SD
    assert smd([2, 2], [0, 2]) == 1.0
    d = dist([1, 2, 3, 4, 5, 6, 7, 8])
    assert d["n"] == 8 and d["median"] == 4.5 and d["min"] == 1 and d["max"] == 8
    assert dist([])["n"] == 0


def test_u1400_audit_job_is_read_only_and_targets_the_control_endpoint():
    t = script()
    assert "#SBATCH --array=0-3" in t
    assert "STEPS=(3889200 3889200 3889200 3889200)" in t
    assert "final_rep_rescue123_h512_s19" in t
    assert "final_rep_rescue123_h512_s22" in t
    assert "must not be mixed into that chain" in t
    ex = "\n".join(l for l in t.splitlines()
                   if l.strip() and not l.lstrip().startswith("#"))
    assert "base123_error_audit.py" in ex
    assert "train_joint_scratch.py" not in ex
    assert '--glove-path "$GLOVE"' in ex
    assert "--allow-glove-fallback" not in ex
    assert "--phase-transition" not in ex
    assert 3_889_200 // (463 * 6) == 1400
