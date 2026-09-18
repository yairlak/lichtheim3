"""Mechanical tests for the frozen V7 pre-lesion validation contract.

Tests that need the V7 checkpoints/heads are marked `needs_artifacts` and skip
cleanly when those files are absent.  Every other test runs here and must pass
BEFORE the design commit.

These tests never inspect scientific model performance.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PKG = os.path.join(ROOT, "paper_programme", "v7_prelesion_validation")
CONTRACT = os.path.join(PKG, "contract")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(PKG, "scripts"))

CANON_TABLE = os.path.join(
    ROOT, "..", "lichtheim3", "outputs", "behavioral_wfe_fulllexicon_93a577f",
    "behavioral_analysis", "tables", "canonical_behavioral_item_table.tsv")
LEXICON = os.path.join(ROOT, "data", "lexicon_en_glove_covered.tsv")

CANON_SHA = "8988aff6fac55ca36cb43ce758f5684f30ae10a6303bdbd7b0b9f462433d5a67"
LEX_SHA = "ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66"


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _manifest(name):
    with open(os.path.join(CONTRACT, name)) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def _contract():
    return json.load(open(os.path.join(CONTRACT, "validation_contract.json")))


def _pool(seed, n=4000, min_len=2, max_len=9):
    from data.phonemes import build_vocab
    v = build_vocab()
    rng = random.Random(seed)
    cons = [v.stoi[s] for s in v.itos[3:] if v.sonority[v.stoi[s]] < 0.9]
    vow = [v.stoi[s] for s in v.itos[3:] if v.sonority[v.stoi[s]] >= 0.95]
    out, seen = [], set()
    while len(out) < n:
        f = []
        for _ in range(rng.randint(1, 3)):
            if rng.random() < 0.85:
                f.append(rng.choice(cons))
            f.append(rng.choice(vow))
            if rng.random() < 0.4:
                f.append(rng.choice(cons))
        if not (min_len <= len(f) <= max_len) or tuple(f) in seen:
            continue
        seen.add(tuple(f))
        out.append(tuple(f))
    return out


# ------------------------------------------------------------ 1. provenance --
def test_canonical_table_sha():
    assert _sha(CANON_TABLE) == CANON_SHA


def test_lexicon_sha_matches_v7_driver_expectation():
    from scripts.naming_comprehension.fresh_ceiling_v7 import EXPECTED
    assert _sha(LEXICON) == LEX_SHA == EXPECTED["lexicon_file_sha256"]


def test_lexicon_population_sizes():
    rows = list(csv.DictReader(open(LEXICON), delimiter="\t"))
    assert len(rows) == 29_571
    assert len({r["word"].strip().lower() for r in rows}) == 29_571
    assert len({" ".join(r["arpabet"].split()) for r in rows}) == 27_981


def test_canonical_table_shape():
    rows = list(csv.DictReader(open(CANON_TABLE), delimiter="\t"))
    assert len(rows) == 14_400 == 1200 * 3 * 4
    assert len({r["item_id"] for r in rows}) == 1200
    by = {}
    for r in rows:
        by.setdefault(r["item_id"], set()).add(r["lichtheim_exposure_status"])
    assert all(len(v) == 1 for v in by.values())


# --------------------------------------------------- 2. frozen state manifest --
def test_state_manifest_has_eight_states():
    m = _manifest("state_manifest.tsv")
    assert len(m) == 8
    assert {r["state_id"] for r in m} == set(_contract()["states"])


def test_post_repair_uses_head_first_c0_only():
    for r in _manifest("state_manifest.tsv"):
        if r["state_kind"] == "POST_REPAIR":
            assert r["repair_head_file"] == "head_first_c0.pt"


def test_head_equality_matches_trace_mechanism():
    """first_c0_theta == final_theta iff the latch coincides with the stop."""
    for r in _manifest("state_manifest.tsv"):
        if r["state_kind"] != "POST_REPAIR":
            continue
        eq = r["head_first_c0_eq_head_final"] == "True"
        assert eq == (int(r["arm_a_first_c0_iter"]) == int(r["arm_a_steps"]))


def test_frozen_source_shas_match_central():
    exp = {"P1": "8e5188055ce5a3fda361b16b482b1010c124473806f07448fe0cd0fd9a7bc707",
           "P2": "8d537e02bf02ad119ec190ba2290991e0000768af8e56085606c3fb7d7efedca",
           "P3": "5f4d4d24c16cebc70dc1f87ad3c113a8696346c75e5121910d1bc2da65a95517",
           "P4": "8ee73263e36d76ba6d815bf57c1d9f6398138280cd24927f64f2bf08e8b7c96a"}
    for r in _manifest("state_manifest.tsv"):
        assert r["source_checkpoint_sha256"] == exp[r["slot"]]


# --------------------------------------------------------- 3. populations ----
def test_primary_manifest_is_378_unique():
    m = _manifest("stimulus_manifest_common_unseen_378.tsv")
    assert len(m) == 378
    assert len({r["item_id"] for r in m}) == 378


def test_exposed_manifest_is_13():
    assert len(_manifest("stimulus_manifest_dorsal_pool_exposed_13.tsv")) == 13


def test_trained_real_exact_is_671():
    assert len(_manifest("stimulus_manifest_trained_real_exact_671.tsv")) == 671


def test_per_seed_manifest_counts():
    m = _manifest("stimulus_manifest_seed_unseen.tsv")
    counts = {}
    for r in m:
        counts[r["slot"]] = counts.get(r["slot"], 0) + 1
    assert counts == {"P1": 384, "P2": 390, "P3": 389, "P4": 387}
    assert len(m) == 1550


def test_primary_plus_exposed_reconstructs_391():
    prim = {r["item_id"] for r in _manifest("stimulus_manifest_common_unseen_378.tsv")}
    exp = {r["item_id"] for r in _manifest("stimulus_manifest_dorsal_pool_exposed_13.tsv")}
    assert not (prim & exp)
    rows = list(csv.DictReader(open(CANON_TABLE), delimiter="\t"))
    nov = {r["item_id"] for r in rows
           if r["lichtheim_exposure_status"] == "NOVEL_PSEUDOWORD"}
    assert len(nov) == 391
    assert prim | exp == nov


def test_primary_has_no_dorsal_pool_overlap_for_any_v7_seed():
    from data.phonemes import build_vocab
    v = build_vocab()
    prim = _manifest("stimulus_manifest_common_unseen_378.tsv")
    ids = {r["item_id"]: tuple(int(x) for x in r["target_phoneme_ids"].split())
           for r in prim}
    for seed in (31, 32, 33, 34):
        pool = set(_pool(seed))
        assert not [i for i, s in ids.items() if s in pool]


def test_primary_has_no_lexicon_phonological_overlap():
    lex = {" ".join(r["arpabet"].split())
           for r in csv.DictReader(open(LEXICON), delimiter="\t")}
    for r in _manifest("stimulus_manifest_common_unseen_378.tsv"):
        assert " ".join(r["target_arpabet"].split()) not in lex


def test_no_oov_phonemes_in_any_population():
    from data.phonemes import build_vocab
    v = build_vocab()
    for name in ("stimulus_manifest_common_unseen_378.tsv",
                 "stimulus_manifest_trained_real_exact_671.tsv",
                 "stimulus_manifest_dorsal_pool_exposed_13.tsv"):
        for r in _manifest(name):
            for tok in r["target_arpabet"].split():
                assert tok in v.stoi


def test_phoneme_ids_agree_with_arpabet():
    from data.phonemes import build_vocab
    v = build_vocab()
    for r in _manifest("stimulus_manifest_common_unseen_378.tsv"):
        want = [v.stoi[t] for t in r["target_arpabet"].split()]
        assert [int(x) for x in r["target_phoneme_ids"].split()] == want
        assert int(r["target_length"]) == len(want)


def test_item_ids_stable_and_unique():
    for name in ("stimulus_manifest_common_unseen_378.tsv",
                 "stimulus_manifest_trained_real_exact_671.tsv"):
        ids = [r["item_id"] for r in _manifest(name)]
        assert len(ids) == len(set(ids))
        assert all(re.fullmatch(r"wfe_\d{4}", i) for i in ids)


def test_dorsal_pools_are_deterministic_and_seed_distinct():
    digs = {s: hashlib.sha256(json.dumps(_pool(s)).encode()).hexdigest()
            for s in (31, 32, 33, 34)}
    assert len(set(digs.values())) == 4
    for s in (31, 32, 33, 34):
        again = hashlib.sha256(json.dumps(_pool(s)).encode()).hexdigest()
        assert again == digs[s]
    frozen = json.load(open(os.path.join(CONTRACT, "population_manifests.json")))
    slot = {31: "P1", 32: "P2", 33: "P3", 34: "P4"}
    for s, d in digs.items():
        assert frozen["pool_digests"][slot[s]]["ordered_sha256"] == d


# ------------------------------------------------------------- 4. free-AR ----
def test_free_ar_horizon_is_global_and_adequate():
    from scripts.naming_comprehension.train_joint_scratch import FREE_AR_MAX_STEPS
    assert FREE_AR_MAX_STEPS == 12 == _contract()["free_ar"]["max_steps"]
    for name in ("stimulus_manifest_common_unseen_378.tsv",
                 "stimulus_manifest_trained_real_exact_671.tsv"):
        for r in _manifest(name):
            assert int(r["target_length"]) + 1 <= FREE_AR_MAX_STEPS


def test_frozen_free_ar_never_consults_target_length_to_terminate():
    """The decode loop must be bounded by the global constant alone."""
    import inspect
    from scripts.naming_comprehension.train_joint_scratch import JointScratchTrainer
    src = inspect.getsource(JointScratchTrainer.free_ar_repetition)
    loop = src.split("for _ in range(")[1].split(")")[0]
    assert loop.strip() == "FREE_AR_MAX_STEPS"
    body = src.split("for _ in range(FREE_AR_MAX_STEPS):")[1].split("for k, f in enumerate(forms)")[0]
    assert "len(f)" not in body and "forms" not in body


def test_our_free_ar_wrapper_matches_frozen_decode_contract():
    import inspect
    import prelesion_eval as pe
    src = inspect.getsource(pe.free_ar_items)
    loop = src.split("for _ in range(")[1].split(")")[0]
    assert loop.strip() == "FREE_AR_MAX_STEPS"
    body = src.split("for _ in range(FREE_AR_MAX_STEPS):")[1].split("for k, e in enumerate(chunk)")[0]
    assert "target" not in body and "len(f)" not in body


def test_levenshtein_and_divergence():
    import prelesion_eval as pe
    assert pe.levenshtein([1, 2, 3], [1, 2, 3]) == 0
    assert pe.levenshtein([1, 2, 3], [1, 2]) == 1
    assert pe.levenshtein([], [1, 2]) == 2
    assert pe.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert pe.first_divergence([1, 9, 3], [1, 2, 3]) == 1
    assert pe.first_divergence([1, 2], [1, 2, 3]) == 2


# ------------------------------------------------------- 5. frozen decisions --
def test_ordering_rule_is_the_frozen_one():
    import prelesion_eval as pe
    assert pe.classify_ordering(0.5, 0.4, 0.2, 0.3)["ordering"] == "WM_DOMINANT"
    assert pe.classify_ordering(0.5, 0.5, 0.2, 0.2)["ordering"] == "TIE"
    assert pe.classify_ordering(0.4, 0.5, 0.3, 0.2)["ordering"] == "LTM_DOMINANT"
    assert pe.classify_ordering(0.5, 0.4, 0.3, 0.2)["ordering"] == "MIXED"
    # boundary: equal on one metric, strictly better on the other
    assert pe.classify_ordering(0.5, 0.5, 0.2, 0.3)["ordering"] == "WM_DOMINANT"


def test_preservation_rule_is_the_frozen_one():
    import prelesion_eval as pe
    assert pe.preservation("WM_DOMINANT", "WM_DOMINANT") == "PRESERVED"
    assert pe.preservation("MIXED", "WM_DOMINANT") == "PRESERVED_FROM_NONDOMINANT_SOURCE"
    assert pe.preservation("WM_DOMINANT", "MIXED") == "NOT_PRESERVED"
    assert pe.preservation("TIE", "LTM_DOMINANT") == "NO_EXPECTED_PATTERN_AT_SOURCE"


def test_contract_forbids_head_final_evaluation():
    c = _contract()
    assert c["head_convention"]["post_repair_scientific_witness"] == "head_first_c0.pt"
    assert c["head_convention"]["head_final_behavior"] == "NOT_SCIENTIFICALLY_EVALUATED"
    assert "EVALUATING_HEAD_FINAL" in c["forbidden"]


def test_summarize_reports_required_quantiles():
    import prelesion_eval as pe
    s = pe.summarize(list(range(101)))
    for k in ("n", "mean", "sd", "min", "p01", "p05", "p25",
              "p50", "p75", "p95", "p99", "max"):
        assert k in s
    assert s["n"] == 101 and s["min"] == 0.0 and s["max"] == 100.0
    assert s["p50"] == 50.0


# ------------------------------------------- 6. artifact-dependent (skipped) --
def _artifacts_present():
    m = _manifest("state_manifest.tsv")
    return all(os.path.exists(r.get("source_checkpoint_path", "")) for r in m)


needs_artifacts = pytest.mark.skipif(
    not _artifacts_present(),
    reason="V7 checkpoints/heads not present on this machine (Jean-Zay /lustre)")


@needs_artifacts
def test_reconstruction_changes_only_head_tensors():
    raise NotImplementedError("runs on the cluster: state_dict diff must be "
                              "exactly {'2.weight','2.bias'}")


@needs_artifacts
def test_checkpoint_and_head_immutable_across_evaluation():
    raise NotImplementedError("runs on the cluster: sha256 before/after")


@needs_artifacts
def test_deterministic_decode_repeat_on_smoke_ids():
    raise NotImplementedError("runs on the cluster: byte-identical repeat")


@needs_artifacts
def test_full_canonical_and_freear_match_frozen_v7_on_smoke():
    raise NotImplementedError("runs on the cluster: parity with frozen battery")
