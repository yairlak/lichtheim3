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

sys.path.insert(0, os.path.join(PKG, "execution"))
import inputs as EXIN  # noqa: E402  machine-portable input resolution

LEXICON = os.path.join(ROOT, "data", "lexicon_en_glove_covered.tsv")

CANON_SHA = EXIN.CANON_TABLE_SHA256
LEX_SHA = "ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66"

needs_canon_table = pytest.mark.skipif(
    not EXIN.canon_table_available(),
    reason=("canonical WFE table not configured on this machine: set "
            "L3_CANON_TABLE (or L3_EXECUTION_INPUTS) to the file with sha256 "
            + CANON_SHA))


def canon_table():
    """SHA-verified path; raises rather than substituting a related table."""
    return EXIN.canon_table_path(required=True)


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
@needs_canon_table
def test_canonical_table_sha():
    assert _sha(canon_table()) == CANON_SHA


def test_lexicon_sha_matches_v7_driver_expectation():
    from scripts.naming_comprehension.fresh_ceiling_v7 import EXPECTED
    assert _sha(LEXICON) == LEX_SHA == EXPECTED["lexicon_file_sha256"]


def test_lexicon_population_sizes():
    rows = list(csv.DictReader(open(LEXICON), delimiter="\t"))
    assert len(rows) == 29_571
    assert len({r["word"].strip().lower() for r in rows}) == 29_571
    assert len({" ".join(r["arpabet"].split()) for r in rows}) == 27_981


@needs_canon_table
def test_canonical_table_shape():
    rows = list(csv.DictReader(open(canon_table()), delimiter="\t"))
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


@needs_canon_table
def test_primary_plus_exposed_reconstructs_391():
    prim = {r["item_id"] for r in _manifest("stimulus_manifest_common_unseen_378.tsv")}
    exp = {r["item_id"] for r in _manifest("stimulus_manifest_dorsal_pool_exposed_13.tsv")}
    assert not (prim & exp)
    rows = list(csv.DictReader(open(canon_table()), delimiter="\t"))
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
    """True only if every P1-P4 SOURCE checkpoint and head_first_c0 resolves
    under L3_V7_RUN_ROOT *and* matches its frozen SHA (fail closed)."""
    return EXIN.artifacts_available()


needs_artifacts = pytest.mark.skipif(
    not _artifacts_present(),
    reason=("V7 SOURCE checkpoints / head_first_c0.pt not resolvable: set "
            "L3_V7_RUN_ROOT to the directory holding "
            "fresh_ceiling_v7_p{1..4}_s{31..34}/"))


@needs_artifacts
@pytest.mark.parametrize("slot", ["P1", "P2", "P3", "P4"])
def test_reconstruction_changes_only_head_tensors(slot):
    """POST_REPAIR must differ from SOURCE in EXACTLY the Arm-A head tensors.

    The acceptable key set is taken from frozen code, never from the outcome:
    `gradient_training_probe.TRAINABLE_NAMES` is Arm-A's own declaration of what
    it optimises, and `frozen_head_probe._isolated_model` loads the derived head
    into `model.ltm.to_semantic`. Any other changed tensor fails the test.
    """
    import torch
    import prelesion_eval as pe
    from scripts.naming_comprehension.gradient_training_probe import TRAINABLE_NAMES

    art = EXIN.resolve_artifacts()[slot]
    glove = _glove_path()

    tr_src, src_model, _ = pe.build_state(art["source_checkpoint"], None, glove)
    src_sd = {k: v.detach().clone() for k, v in src_model.state_dict().items()}

    _, post_model, prov = pe.build_state(
        art["source_checkpoint"], art["repair_head"], glove)
    post_sd = post_model.state_dict()

    assert set(src_sd) == set(post_sd), "architecture changed during reconstruction"
    changed = {k for k in src_sd if not torch.equal(src_sd[k], post_sd[k])}
    assert changed == set(TRAINABLE_NAMES), (
        f"{slot}: expected exactly {sorted(TRAINABLE_NAMES)} to change, "
        f"got {sorted(changed)}")

    # the deployed head digest recorded must match the frozen one
    assert prov["repair_head_deployed_state_sha256"] == \
        art["repair_head_deployed_state_sha256"]


@needs_artifacts
@pytest.mark.parametrize("slot", ["P1", "P2", "P3", "P4"])
def test_checkpoint_and_head_immutable_across_evaluation(slot):
    """Byte-level immutability of SOURCE checkpoint and head across a real
    (small) evaluation, not merely across a load."""
    import prelesion_eval as pe

    art = EXIN.resolve_artifacts()[slot]
    ck, hd = art["source_checkpoint"], art["repair_head"]
    ck_before, hd_before = pe.sha256_file(ck), pe.sha256_file(hd)

    tr, model, _ = pe.build_state(ck, hd, _glove_path())
    pe.free_ar_items(tr, model, _smoke_entries(tr), routes=("full",))

    assert pe.sha256_file(ck) == ck_before == art["source_checkpoint_sha256"]
    assert pe.sha256_file(hd) == hd_before == art["repair_head_file_sha256"]
    pe.assert_source_unchanged(ck, ck_before)


@needs_artifacts
def test_deterministic_decode_repeat_on_smoke_ids():
    """Two identical decodes of the predeclared smoke set must agree exactly.

    Smoke items are fixed in `execution/inputs.py` before any execution and are
    chosen without reference to any scientific outcome (lowest bank indices).
    No stochastic code is involved.
    """
    import prelesion_eval as pe

    art = EXIN.resolve_artifacts()["P1"]
    tr, model, _ = pe.build_state(
        art["source_checkpoint"], art["repair_head"], _glove_path())
    ents = _smoke_entries(tr)
    state_before = pe.model_state_digest(model)

    a = pe.free_ar_items(tr, model, ents, routes=("full", "wm", "ltm"))
    b = pe.free_ar_items(tr, model, ents, routes=("full", "wm", "ltm"))
    assert _digest(a) == _digest(b), "decode is not deterministic"
    assert pe.model_state_digest(model) == state_before, \
        "model state_dict changed during evaluation"


@needs_artifacts
def test_full_canonical_and_freear_match_frozen_v7_on_smoke():
    """Evaluator parity between the new read-only wrapper and frozen V7.

    What is tested, precisely:
      * free-AR: our item-level wrapper, aggregated to FULL exact-match, must
        equal the frozen `JointScratchTrainer.free_ar_repetition` aggregate on
        the same predeclared smoke items and the same model. This is an
        equivalence test between wrapper and frozen evaluator -- no archived
        item-level output is fabricated.
      * canonical: the contract reuses the frozen `repetition_snapshot`
        unchanged (no wrapper), so parity is asserted as reproducibility of the
        frozen function on the smoke set, and its FULL readout is required to be
        a well-formed rate.
    The frozen OFFICIAL AGGREGATE battery outcomes are retained separately, as
    an external invariant for the full scientific run
    (`EXIN.FROZEN_OFFICIAL_POST_BATTERY`); they are deliberately NOT asserted
    here, because this smoke subset is not the official population.
    """
    import prelesion_eval as pe
    from scripts.naming_comprehension.train_tasks import repetition_snapshot

    art = EXIN.resolve_artifacts()["P1"]
    tr, model, _ = pe.build_state(
        art["source_checkpoint"], art["repair_head"], _glove_path())
    idx = EXIN.SMOKE_REAL_BANK_INDICES
    ents = _smoke_entries(tr)

    # -- genuine free-AR parity --------------------------------------------
    orig = tr.model
    try:
        tr.model = model
        frozen_far = tr.free_ar_repetition(idx, routes=("full", "wm", "ltm"))
    finally:
        tr.model = orig
    ours = pe.free_ar_items(tr, model, ents, routes=("full", "wm", "ltm"))
    for route in ("full", "wm", "ltm"):
        mine = sum(r["exact"] for r in ours[route]) / len(idx)
        assert mine == pytest.approx(frozen_far[route], abs=0.0), (
            f"free-AR wrapper disagrees with frozen evaluator on {route}: "
            f"{mine} vs {frozen_far[route]}")

    # -- canonical (forced-length) reproducibility --------------------------
    s1 = repetition_snapshot(model, tr.vocab, tr.entries, idx, tr.bank_raw,
                             "cpu", include_teacher_forced=False)
    s2 = repetition_snapshot(model, tr.vocab, tr.entries, idx, tr.bank_raw,
                             "cpu", include_teacher_forced=False)
    e1 = s1["primary_readout"]["exact_match"]
    e2 = s2["primary_readout"]["exact_match"]
    for route in ("full", "wm", "ltm"):
        assert float(e1[route]) == float(e2[route])
        assert 0.0 <= float(e1[route]) <= 1.0


# ------------------------------------------------- artifact-test helpers ----
def _glove_path():
    from scripts.naming_comprehension.ceiling_source_completion import GLOVE
    return GLOVE


def _smoke_entries(tr):
    """Predeclared smoke entries as the evaluator consumes them."""
    return [{"item_id": f"bank_{i}", "phonemes": list(tr.entries[i].phonemes)}
            for i in EXIN.SMOKE_REAL_BANK_INDICES]


def _digest(out):
    import hashlib
    import json
    return hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()


# ------------------------------------- static proof the above are implemented --
ARTIFACT_TEST_NAMES = (
    "test_reconstruction_changes_only_head_tensors",
    "test_checkpoint_and_head_immutable_across_evaluation",
    "test_deterministic_decode_repeat_on_smoke_ids",
    "test_full_canonical_and_freear_match_frozen_v7_on_smoke",
)


def test_no_placeholder_bodies_remain_in_suite():
    """No test function may still be a NotImplementedError placeholder.

    Checked over function BODIES via AST, so this test's own reference to the
    sentinel name does not trip it.
    """
    import ast
    src = open(__file__).read()
    tree = ast.parse(src)
    offenders = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.FunctionDef) or not n.name.startswith("test_"):
            continue
        if n.name == "test_no_placeholder_bodies_remain_in_suite":
            continue
        for st in ast.walk(n):
            if (isinstance(st, ast.Raise) and st.exc is not None
                    and "NotImplementedError" in ast.dump(st.exc)):
                offenders.append(n.name)
    assert not offenders, f"placeholder test bodies remain: {sorted(set(offenders))}"


def test_evaluator_module_has_no_placeholders():
    import ast
    import prelesion_eval as pe
    src = open(pe.__file__).read()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Raise) and n.exc is not None:
            assert "NotImplementedError" not in ast.dump(n.exc)


def test_artifact_dependent_tests_are_really_implemented():
    """Each artifact test must have a real body that exercises the evaluator."""
    import ast
    import inspect
    tree = ast.parse(open(__file__).read())
    fns = {n.name: n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef)}
    for name in ARTIFACT_TEST_NAMES:
        assert name in fns, f"{name} missing"
        body = ast.get_source_segment(open(__file__).read(), fns[name])
        assert "prelesion_eval" in body, f"{name} does not use the evaluator"
        for st in ast.walk(fns[name]):
            assert not (isinstance(st, ast.Raise) and st.exc is not None
                        and "NotImplementedError" in ast.dump(st.exc))
        # more than a docstring + pass
        stmts = [st for st in fns[name].body
                 if not (isinstance(st, ast.Expr)
                         and isinstance(st.value, ast.Constant))]
        assert len(stmts) >= 3, f"{name} body is too thin to be a real test"


def test_artifact_guard_uses_real_resolution():
    """The skip guard must consult real SHA-verified resolution, not a field
    that does not exist in the frozen manifest (the earlier defect)."""
    import inspect
    src = inspect.getsource(_artifacts_present)
    assert "artifacts_available" in src
    assert "source_checkpoint_path" not in src
    cols = set(_manifest("state_manifest.tsv")[0])
    assert "source_checkpoint_path" not in cols
    assert {"source_checkpoint_sha256", "repair_head_file_sha256"} <= cols


def test_artifact_paths_derive_from_frozen_layout():
    d = EXIN.derive_artifact_paths("/RUNS", "P1", 31, 4125330, 1485)
    assert d["run_id"] == "fresh_ceiling_v7_p1_s31"
    assert d["source_checkpoint"] == \
        "/RUNS/fresh_ceiling_v7_p1_s31/checkpoints/step_04125330.pt"
    assert d["repair_head"] == \
        "/RUNS/fresh_ceiling_v7_p1_s31/post/seed31_u1485_A/head_first_c0.pt"


def test_input_resolution_fails_closed_on_sha_mismatch(tmp_path):
    bad = tmp_path / "canonical_behavioral_item_table.tsv"
    bad.write_text("not the frozen table\n")
    os.environ["L3_CANON_TABLE"] = str(bad)
    try:
        with pytest.raises(EXIN.InputResolutionError) as e:
            EXIN.canon_table_path(required=True)
        assert "SHA256 mismatch" in str(e.value)
    finally:
        del os.environ["L3_CANON_TABLE"]


def test_input_resolution_fails_closed_on_missing_file():
    os.environ["L3_CANON_TABLE"] = "/nonexistent/table.tsv"
    try:
        with pytest.raises(EXIN.InputResolutionError):
            EXIN.canon_table_path(required=True)
    finally:
        del os.environ["L3_CANON_TABLE"]


def test_frozen_official_battery_invariants_recorded():
    """The external aggregate invariant must match the frozen contract."""
    inv = EXIN.FROZEN_OFFICIAL_POST_BATTERY
    assert inv["P4"] == {"Rcan": 1, "Rfree": 1, "N": 0, "C": 0}
    for slot in ("P1", "P2", "P3"):
        assert inv[slot] == {"Rcan": 0, "Rfree": 0, "N": 0, "C": 0}
    for r in _manifest("state_manifest.tsv"):
        if r["state_kind"] != "POST_REPAIR":
            continue
        want = inv[r["slot"]]
        assert r["frozen_official_post_battery_Rcan_Rfree_N_C"] == \
            f"{want['Rcan']}/{want['Rfree']}/{want['N']}/{want['C']}"


def test_artifact_resolution_fails_closed_when_root_set_but_empty(tmp_path):
    """A configured-but-wrong run root must RAISE, never silently skip.

    This is the defect class that made the old guard vacuous: it must be
    impossible for a misconfigured root to look like "artifacts absent".
    """
    os.environ["L3_V7_RUN_ROOT"] = str(tmp_path)
    try:
        with pytest.raises(EXIN.InputResolutionError) as e:
            EXIN.resolve_artifacts(required=True)
        assert "does not exist" in str(e.value)
        # and the boolean helper reports False rather than raising
        assert EXIN.artifacts_available() is False
    finally:
        del os.environ["L3_V7_RUN_ROOT"]


def test_artifact_resolution_rejects_nondirectory_root(tmp_path):
    f = tmp_path / "not_a_dir"
    f.write_text("x")
    os.environ["L3_V7_RUN_ROOT"] = str(f)
    try:
        with pytest.raises(EXIN.InputResolutionError):
            EXIN.v7_run_root(required=True)
    finally:
        del os.environ["L3_V7_RUN_ROOT"]


DESIGN_COMMIT = "2d240e1fd4b81ca93b4144b772eb21e8700d8ddf"
FROZEN_CONTRACT_FILES = (
    "V7_INTACT_ROUTE_VALIDATION_CONTRACT.md", "validation_contract.json",
    "population_manifests.json", "state_manifest.tsv",
    "stimulus_manifest_common_unseen_378.tsv",
    "stimulus_manifest_dorsal_pool_exposed_13.tsv",
    "stimulus_manifest_seed_unseen.tsv",
    "stimulus_manifest_trained_real_exact_671.tsv",
)


def test_contract_is_byte_identical_to_design_commit():
    """The scientific contract may never drift from the preregistration anchor."""
    import subprocess
    for name in FROZEN_CONTRACT_FILES:
        rel = f"paper_programme/v7_prelesion_validation/contract/{name}"
        p = subprocess.run(["git", "show", f"{DESIGN_COMMIT}:{rel}"],
                           capture_output=True, cwd=ROOT)
        assert p.returncode == 0, f"{rel} missing at {DESIGN_COMMIT}"
        frozen = hashlib.sha256(p.stdout).hexdigest()
        here = hashlib.sha256(open(os.path.join(ROOT, rel), "rb").read()).hexdigest()
        assert here == frozen, f"CONTRACT DRIFT in {name}"


def test_no_results_namespace_exists():
    assert not os.path.exists(os.path.join(PKG, "results"))


def test_no_training_or_lesion_code_added():
    """The execution plumbing must stay read-only in kind."""
    import ast
    banned = ("backward", "optimizer", "requires_grad_", "load_state_dict",
              "save", "lesion", "p_max")
    for rel in ("execution/inputs.py", "scripts/prelesion_eval.py"):
        src = open(os.path.join(PKG, rel)).read()
        tree = ast.parse(src)
        called = {n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        for b in banned:
            assert b not in called, f"{rel} calls banned operation {b}()"
        assert "torch.save" not in src and "def train" not in src
