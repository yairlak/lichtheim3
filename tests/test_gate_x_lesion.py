"""T1-T12 — required implementation tests for GATE x LESION / RECOVERY.

Every test that touches a real checkpoint uses a deliberately tiny, non-canonical,
quarantined subset.  **No non-zero-severity canonical lesion is run anywhere in this
file.**  The only full-population work is T2, the severity-0 authoritative intact null,
which is explicitly permitted as an intact control.

Run:  python3 -m pytest tests/test_gate_x_lesion.py -v
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gate_x_lesion.evaluate import collect_item_level_lesioned
from gate_x_lesion.hooks import lesioned_route, state_dict_sha256
from gate_x_lesion.noise import (EpsilonCache, FROZEN_LAMBDAS, amplitude,
                                 base_epsilon, eta, identity_json)
from gate_x_lesion.outcomes import (OUTCOME_A, OUTCOME_B, OUTCOME_D, OUTCOME_F,
                                    OUTCOME_UNDETERMINED, SeedRecord, SeverityVerdict,
                                    classify_convention, classify_joint,
                                    classify_route_family, evaluate_severity)
from gate_x_lesion.sd import measure_intact_sd
from gate_x_lesion.targets import FROZEN_ROUTES, assert_site_compatible
from scripts.gate_x_lesion.run_gate_x_lesion import (QUARANTINE_DIR, SMOKE_SUFFIX,
                                                     assert_execution_authorised,
                                                     assert_full_population,
                                                     assert_quarantined)

REPO = "/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3"
OUT_BASE = os.path.join(ROOT, "paper_programme", "gate_x_lesion_recovery")
MANIFEST = os.path.join(OUT_BASE, "checkpoint_manifest.proposed.tsv")
GATING = os.path.join(REPO, "wt-gating-diagnostics", "paper_programme",
                      "gating_route_diagnostics")

#: Deliberately tiny and non-canonical.  Never a scientific population.
SMOKE_N = 24
SMOKE_LAMBDA = 0.25
SMOKE_SEED = 0


def _manifest_rows():
    with open(MANIFEST, newline="") as f:
        return [r for r in csv.DictReader(f, delimiter="\t") if r["in_scope"] == "1"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


@pytest.fixture(scope="module")
def states():
    """Both in-scope states, reconstructed once by the frozen GATING recipe."""
    from scripts.gating_diagnostics.run_gate_route_audit import build_state
    out = {}
    for row in _manifest_rows():
        legacy = {"state_id": row["state_id"], "arm": row["arm"],
                  "artifact_path": row["base_artifact_path"],
                  "artifact_sha256": row["base_artifact_sha256"],
                  "applies_head_path": row["applies_head_path"],
                  "applies_head_sha256": row["applies_head_sha256"],
                  "source_u": row["source_u"]}
        tr, model, prov, before, ckpt = build_state(legacy, "cpu")
        out[row["state_id"]] = dict(row=row, tr=tr, model=model, prov=prov, ckpt=ckpt)
    return out


# ===========================================================================  T1

def test_T1_exact_checkpoint_hash_resolution():
    """W3/W4 resolve exactly; computed SHA256 matches the frozen GATING manifest."""
    frozen = {r["state_id"]: r for r in csv.DictReader(
        open(os.path.join(GATING, "checkpoint_manifest.tsv"), newline=""),
        delimiter="\t")}
    rows = _manifest_rows()
    assert {r["state_id"] for r in rows} == {"W3_REP", "W4_REP"}

    for r in rows:
        f = frozen[r["state_id"]]
        # no substitution: our manifest quotes the frozen one verbatim
        assert r["base_artifact_path"] == f["artifact_path"]
        assert r["base_artifact_sha256"] == f["artifact_sha256"]
        assert r["applies_head_path"] == f["applies_head_path"]
        assert r["applies_head_sha256"] == f["applies_head_sha256"]

        base = os.path.join(REPO, r["base_artifact_path"])
        head = os.path.join(REPO, r["applies_head_path"])
        assert os.path.exists(base), base
        assert os.path.exists(head), head
        assert sha256_file(base) == r["base_artifact_sha256"]
        assert sha256_file(head) == r["applies_head_sha256"]

        # the reconstruction recipe is preserved, not flattened to one file
        assert r["state_kind"] == "RECONSTRUCTION_RECIPE_base_plus_head"
        composite = hashlib.sha256(
            ("gxlr-state-v1|" + r["base_artifact_sha256"] + "|"
             + r["applies_head_sha256"]).encode()).hexdigest()
        assert r["state_sha256"] == composite


def test_T1b_head_carries_exactly_the_final_layer(states):
    for sid, st in states.items():
        head = torch.load(os.path.join(REPO, st["row"]["applies_head_path"]),
                          map_location="cpu", weights_only=False)
        assert set(head["state"]) == {"2.weight", "2.bias"}, sid


# ===========================================================================  T2

def test_T2_severity_zero_authoritative_null(states):
    """Severity 0 reproduces the authoritative intact GATING result.

    Both selected states, both decoding conventions, FULL canonical population.
    This is an intact control; no non-zero lesion is run here.
    """
    for sid, st in states.items():
        model, tr = st["model"], st["tr"]
        before = state_dict_sha256(model)
        indices = list(range(len(tr.entries)))
        assert len(indices) == 29571

        # batch_size MUST be 256, the value the frozen GATING record was produced
        # with.  Both encoders pack their input (`pack_padded_sequence`), and the
        # packed GRU kernel's reduction order depends on batch composition, so a
        # different batch size moves the *gate* by 1-2 ulp on a handful of items
        # (measured: 15/2048 at batch 512, max 1.19e-07).  It changes no
        # prediction and no epsilon — epsilon is per-item and batch-invariant by
        # construction — but the gate is a reported continuous measurement, so
        # the batch size is pinned for exact comparability.  See IMPLEMENTATION_NOTES.md.
        rows = collect_item_level_lesioned(model, tr.vocab, tr.entries, indices,
                                           "cpu", hook=None, batch_size=256,
                                           free_ar=True)
        assert state_dict_sha256(model) == before, f"{sid}: checkpoint mutated"

        frozen = list(csv.DictReader(
            open(os.path.join(GATING, "figure_source_data",
                              f"item_level_{sid}.tsv"), newline=""), delimiter="\t"))
        assert len(rows) == len(frozen) == 29571

        for conv in ("canonical", "freear"):
            disc = sum(r[f"{conv}_discordant_prediction"] for r in rows)
            assert disc == 0, f"{sid}/{conv}: {disc} NATIVE-vs-FIXED05 discordant items"

        # identical to the frozen authoritative record, item by item
        for a, b in zip(rows, frozen):
            assert a["word"] == b["word"]
            for conv in ("canonical", "freear"):
                for route in ("full", "wm", "ltm", "fixed05"):
                    assert a[f"{conv}_{route}_predicted"] == b[f"{conv}_{route}_predicted"], (
                        f"{sid} {a['word']} {conv}/{route}")
            assert abs(a["gate"] - float(b["gate"])) == 0.0


# ===========================================================================  T3

def test_T3_base_noise_reproducible_in_process():
    kw = dict(state_sha256="a" * 64, route="wm_encoder_state", lesion_seed=2,
              item_id="cat")
    a = base_epsilon(n_units=128, **kw)
    b = base_epsilon(n_units=128, **kw)
    assert torch.equal(a, b)
    assert a.dtype == torch.float32 and a.shape == (128,)
    assert float(a.min()) >= -1.0 and float(a.max()) < 1.0

    # identity actually discriminates
    for change in (dict(state_sha256="b" * 64), dict(route="ltm_encoder_state"),
                   dict(lesion_seed=3), dict(item_id="dog")):
        assert not torch.equal(a, base_epsilon(n_units=128, **{**kw, **change}))

    # cache hit is bitwise indistinguishable from a cold compute
    c = EpsilonCache("a" * 64, "wm_encoder_state", 2, 128)
    assert torch.equal(c.epsilon("cat"), a)
    assert torch.equal(c.epsilon("cat"), a)
    assert c.n_computed == 1 and c.n_served == 2


def test_T3b_base_noise_reproducible_across_processes():
    """A fresh interpreter must reconstruct the identical tensor."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from gate_x_lesion.noise import base_epsilon\n"
        "e = base_epsilon('a'*64, 'wm_encoder_state', 2, 'cat', 128)\n"
        "import hashlib; print(hashlib.sha256(e.numpy().tobytes()).hexdigest())\n"
        % ROOT)
    env = dict(os.environ)
    digests = set()
    for salt in ("0", "1", "12345"):
        env["PYTHONHASHSEED"] = salt
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env, cwd=ROOT)
        assert r.returncode == 0, r.stderr
        digests.add(r.stdout.strip())
    local = hashlib.sha256(
        base_epsilon("a" * 64, "wm_encoder_state", 2, "cat", 128).numpy().tobytes()
    ).hexdigest()
    assert digests == {local}, (
        "epsilon is not process-stable — PYTHONHASHSEED changed the result")


def test_T3c_builtin_hash_is_not_used_for_rng_identity():
    src = open(os.path.join(ROOT, "gate_x_lesion", "noise.py")).read()
    body = "\n".join(l for l in src.splitlines()
                     if not l.strip().startswith("#") and "hash()" not in l)
    assert "hash(" not in body.replace("hashlib.sha256(", "").replace(
        "identity_digest(", "").replace("_digest(", "")


# ===========================================================================  T4

def test_T4_lambda_excluded_from_rng_identity():
    """The same base identity yields the same epsilon at 0.25 / 0.50 / 1.00."""
    kw = dict(state_sha256="c" * 64, route="ltm_encoder_state", lesion_seed=1,
              item_id="house")
    cache = EpsilonCache(kw["state_sha256"], kw["route"], kw["lesion_seed"], 512)
    base = base_epsilon(n_units=512, **kw)
    for lam in FROZEN_LAMBDAS:
        assert torch.equal(cache.epsilon(kw["item_id"]), base), lam
    assert cache.n_computed == 1, "epsilon was redrawn across severities"

    # structurally: the serialised identity has exactly four keys, none of them lambda
    payload = json.loads(identity_json(**kw))
    assert set(payload) == {"state_sha256", "route", "lesion_seed", "item_id"}
    for forbidden in ("lambda", "lam", "severity", "fusion", "convention"):
        assert forbidden not in payload


# ===========================================================================  T5

def test_T5_nested_scaling_is_bitwise_exact():
    """eta(0.50)==2*eta(0.25), eta(1.00)==2*eta(0.50), eta(1.00)==4*eta(0.25).

    Frozen deterministic tolerance: 0.0 (bitwise).  Justified because the frozen
    severities are exact binary powers of two.  Asserted on the exact implementation
    path (`eta`), not on a reimplementation.
    """
    for sd in (0.30000001192092896, 0.1234567, 1.0, 7.5e-3, 2.71828):
        e = base_epsilon("d" * 64, "wm_encoder_state", 0, f"w{sd}", 256)
        e25, e50, e100 = (eta(e, lam, sd) for lam in FROZEN_LAMBDAS)
        assert torch.equal(e50, 2.0 * e25), sd
        assert torch.equal(e100, 2.0 * e50), sd
        assert torch.equal(e100, 4.0 * e25), sd
        # same direction, magnitude only
        nz = e25 != 0
        assert torch.equal(torch.sign(e25[nz]), torch.sign(e100[nz]))
        assert amplitude(0.50, sd) == 2.0 * amplitude(0.25, sd)
        assert amplitude(1.00, sd) == 2.0 * amplitude(0.50, sd)


def test_T5b_zero_amplitude_is_exactly_zero():
    e = base_epsilon("d" * 64, "wm_encoder_state", 0, "x", 32)
    assert torch.equal(eta(e, 0.0, 0.5), torch.zeros(32))
    assert torch.equal(eta(e, 0.25, 0.0), torch.zeros(32))   # ZERO_SD_BEHAVIOR


# ===========================================================================  T6

def test_T6_native_and_fixed05_consume_the_same_tensor(states):
    """Quarantined non-canonical subset: NATIVE and FIXED05 see identical eta."""
    st = states["W3_REP"]
    model, tr = st["model"], st["tr"]
    sd_info = measure_intact_sd(model, tr.vocab, tr.entries,
                                routes=("wm_encoder_state",), n_items=256)
    sd = sd_info["wm_encoder_state"]["sd"]
    words = [tr.entries[i].word for i in range(SMOKE_N)]

    seen = []
    with lesioned_route(model, route="wm_encoder_state",
                        state_sha256=st["row"]["state_sha256"],
                        lesion_seed=SMOKE_SEED, lam=SMOKE_LAMBDA, sd=sd) as hook:
        hook.bind(words)
        captured = hook._eta.clone()
        # both fusion conditions are read from ONE route="full" forward
        from gating_diagnostics.gate_probe import _route_step_logits, fixed_mix_logits
        from evaluate.hooks import make_batch
        b = make_batch([list(tr.entries[i].phonemes) for i in range(SMOKE_N)],
                       tr.vocab, "cpu")
        dec = b["enc_in"].new_full((SMOKE_N, 1), tr.vocab.bos_id)
        for route in ("full", "fixed05"):
            _route_step_logits(model, b["enc_in"], b["enc_mask"], dec, route)
            seen.append(hook._eta.clone())

    assert torch.equal(seen[0], seen[1]), "NATIVE and FIXED05 saw different eta"
    assert torch.equal(seen[0], captured)

    # and the tensor is exactly epsilon * (lambda * SD) for those items
    cache = EpsilonCache(st["row"]["state_sha256"], "wm_encoder_state", SMOKE_SEED, 128)
    assert torch.equal(captured, cache.batch_epsilon(words) * amplitude(SMOKE_LAMBDA, sd))


# ===========================================================================  T7

def test_T7_per_item_frozen_across_decoder_steps(states):
    """Every encoder call during one item's AR decode reuses the same tensor."""
    st = states["W3_REP"]
    model, tr = st["model"], st["tr"]
    sd = measure_intact_sd(model, tr.vocab, tr.entries,
                           routes=("ltm_encoder_state",),
                           n_items=256)["ltm_encoder_state"]["sd"]
    words = [tr.entries[i].word for i in range(SMOKE_N)]
    indices = list(range(SMOKE_N))

    snapshots = []
    with lesioned_route(model, route="ltm_encoder_state",
                        state_sha256=st["row"]["state_sha256"],
                        lesion_seed=SMOKE_SEED, lam=SMOKE_LAMBDA, sd=sd) as hook:
        orig = hook.__class__.__call__

        def spy(self, module, inputs, output):
            snapshots.append(self._eta.clone())
            return orig(self, module, inputs, output)

        hook.__class__.__call__ = spy
        try:
            collect_item_level_lesioned(model, tr.vocab, tr.entries, indices, "cpu",
                                        hook=hook, batch_size=SMOKE_N, free_ar=True)
        finally:
            hook.__class__.__call__ = orig

    assert len(snapshots) > 5, "encoder was not re-run across decoder steps"
    for s in snapshots[1:]:
        assert torch.equal(s, snapshots[0]), "lesion tensor changed mid-item"
    # one build for the batch, many hook firings
    assert hook.n_eta_builds == 1
    assert hook.n_calls == len(snapshots)


# ===========================================================================  T8

def test_T8_dorsal_structural_null(states):
    """Dorsal perturbation: isolated WM may change; c_LTM, g and isolated LTM may not.

    Quarantined non-canonical subset.  A failure here is an implementation/wiring
    failure, NOT a scientific result.
    """
    st = states["W4_REP"]
    model, tr = st["model"], st["tr"]
    sd = measure_intact_sd(model, tr.vocab, tr.entries,
                           routes=("wm_encoder_state",),
                           n_items=256)["wm_encoder_state"]["sd"]
    indices = list(range(SMOKE_N))
    base = collect_item_level_lesioned(model, tr.vocab, tr.entries, indices, "cpu",
                                       hook=None, batch_size=SMOKE_N, free_ar=True)

    with lesioned_route(model, route="wm_encoder_state",
                        state_sha256=st["row"]["state_sha256"],
                        lesion_seed=SMOKE_SEED, lam=1.0, sd=sd) as hook:
        les = collect_item_level_lesioned(model, tr.vocab, tr.entries, indices, "cpu",
                                          hook=hook, batch_size=SMOKE_N, free_ar=True)

    for a, b in zip(base, les):
        assert a["c_LTM"] == b["c_LTM"], f"{a['word']}: dorsal lesion moved c_LTM"
        assert a["gate"] == b["gate"], f"{a['word']}: dorsal lesion moved g"
        for conv in ("canonical", "freear"):
            assert a[f"{conv}_ltm_predicted"] == b[f"{conv}_ltm_predicted"], (
                f"{a['word']}: dorsal lesion changed isolated LTM ({conv})")
    # the intervention is not inert: isolated WM is allowed to change, and does
    changed = sum(1 for a, b in zip(base, les)
                  if a["canonical_wm_predicted"] != b["canonical_wm_predicted"])
    assert changed > 0, "dorsal lesion had no effect on isolated WM — site is inert"


# ===========================================================================  T9

def test_T9_ventral_reachability_and_route_isolation(states):
    """Ventral perturbation reaches ventral state, c/g; dorsal route-only is untouched.

    Wiring test only.  Not an effect-size result.
    """
    st = states["W4_REP"]
    model, tr = st["model"], st["tr"]
    sd = measure_intact_sd(model, tr.vocab, tr.entries,
                           routes=("ltm_encoder_state",),
                           n_items=256)["ltm_encoder_state"]["sd"]
    indices = list(range(SMOKE_N))
    base = collect_item_level_lesioned(model, tr.vocab, tr.entries, indices, "cpu",
                                       hook=None, batch_size=SMOKE_N, free_ar=True)

    with lesioned_route(model, route="ltm_encoder_state",
                        state_sha256=st["row"]["state_sha256"],
                        lesion_seed=SMOKE_SEED, lam=1.0, sd=sd) as hook:
        les = collect_item_level_lesioned(model, tr.vocab, tr.entries, indices, "cpu",
                                          hook=hook, batch_size=SMOKE_N, free_ar=True)

    for a, b in zip(base, les):
        for conv in ("canonical", "freear"):
            assert a[f"{conv}_wm_predicted"] == b[f"{conv}_wm_predicted"], (
                f"{a['word']}: ventral lesion changed dorsal route-only output ({conv})")
    assert any(a["c_LTM"] != b["c_LTM"] for a, b in zip(base, les)), "c_LTM unreachable"
    assert any(a["gate"] != b["gate"] for a, b in zip(base, les)), "g unreachable"
    assert any(a["canonical_ltm_predicted"] != b["canonical_ltm_predicted"]
               for a, b in zip(base, les)), "ventral output unreachable"


def test_T9b_ventral_site_refuses_wrong_encoder_mode(states):
    """The silent-no-op hazard is guarded, not merely documented."""
    st = states["W3_REP"]
    model = st["model"]
    assert_site_compatible(model, "ltm_encoder_state")          # passes as configured
    old = model.ltm.cfg.ltm_encoder_mode
    try:
        object.__setattr__(model.ltm.cfg, "ltm_encoder_mode", "bigru_masked_mean")
        with pytest.raises(RuntimeError, match="SILENT NO-OP"):
            assert_site_compatible(model, "ltm_encoder_state")
    finally:
        object.__setattr__(model.ltm.cfg, "ltm_encoder_mode", old)


# ==========================================================================  T10

def test_T10_checkpoint_restoration(states):
    """state_dict hash identical before and after every lesion context, incl. on error."""
    for sid, st in states.items():
        model, tr = st["model"], st["tr"]
        before = state_dict_sha256(model)
        sd_info = measure_intact_sd(model, tr.vocab, tr.entries, n_items=256)
        assert state_dict_sha256(model) == before, f"{sid}: SD measurement mutated state"

        indices = list(range(SMOKE_N))
        for route in FROZEN_ROUTES:
            for lam in FROZEN_LAMBDAS:
                with lesioned_route(model, route=route,
                                    state_sha256=st["row"]["state_sha256"],
                                    lesion_seed=SMOKE_SEED, lam=lam,
                                    sd=sd_info[route]["sd"]) as hook:
                    collect_item_level_lesioned(model, tr.vocab, tr.entries, indices,
                                                "cpu", hook=hook, batch_size=SMOKE_N,
                                                free_ar=False)
                assert state_dict_sha256(model) == before, f"{sid}/{route}/{lam}"

        # hooks are removed even when the body raises
        n_hooks = len(model.wm.encoder._forward_hooks)
        with pytest.raises(ValueError):
            with lesioned_route(model, route="wm_encoder_state",
                                state_sha256=st["row"]["state_sha256"],
                                lesion_seed=0, lam=1.0,
                                sd=sd_info["wm_encoder_state"]["sd"]):
                raise ValueError("boom")
        assert len(model.wm.encoder._forward_hooks) == n_hooks
        assert state_dict_sha256(model) == before

        # the file on disk is untouched
        assert sha256_file(st["ckpt"]) == st["row"]["base_artifact_sha256"]


# ==========================================================================  T11

def test_T11_quarantine_enforced():
    """Smoke outputs cannot escape quarantine or share the scientific namespace."""
    # a truncated run may not write to full-result paths
    with pytest.raises(RuntimeError, match="HARD STOP"):
        assert_full_population(400, smoke=False)
    assert_full_population(400, smoke=True) is None
    assert_full_population(None, smoke=False) is None

    # non-zero severity on the full population needs CENTRAL go
    with pytest.raises(RuntimeError, match="GO_FOR_SCIENTIFIC_EXECUTION = NO"):
        assert_execution_authorised([0.25], smoke=False, central_go=False)
    assert_execution_authorised([0.0], smoke=False, central_go=False) is None
    assert_execution_authorised([0.25], smoke=True, central_go=False) is None

    # smoke paths must contain the marker and carry the suffix
    q = os.path.join(OUT_BASE, QUARANTINE_DIR)
    assert QUARANTINE_DIR in q
    assert_quarantined(os.path.join(q, f"summary_W3{SMOKE_SUFFIX}.json"), smoke=True)
    with pytest.raises(RuntimeError, match="HARD STOP"):
        assert_quarantined(os.path.join(OUT_BASE, "scientific_execution", "x.json"),
                           smoke=True)
    with pytest.raises(RuntimeError, match=SMOKE_SUFFIX):
        assert_quarantined(os.path.join(q, "summary.json"), smoke=True)
    # and the two namespaces are disjoint
    assert not os.path.realpath(q).startswith(
        os.path.realpath(os.path.join(OUT_BASE, "scientific_execution")) + os.sep)


def test_T11b_quarantined_artifacts_declare_themselves():
    """If a smoke run has been executed, its artifacts must be self-marking."""
    q = os.path.join(OUT_BASE, QUARANTINE_DIR)
    if not os.path.isdir(q):
        pytest.skip("no smoke artifacts present")
    found = 0
    for dirpath, _, files in os.walk(q):
        for fn in files:
            p = os.path.join(dirpath, fn)
            if fn.endswith((".json", ".tsv")):
                found += 1
                assert SMOKE_SUFFIX in fn, p
                assert QUARANTINE_DIR in p, p
                head = open(p).read(4096)
                assert "NOT_SCIENTIFIC_RESULT" in head, p
    assert found > 0


# ==========================================================================  T12

def _rec(lam, seed, *, net, p, changed=100, acc=0.6, modal=0.1):
    return SeedRecord(lam=lam, seed=seed, n_changed_vs_intact=changed,
                      native_exact_match=acc, modal_prediction_share=modal,
                      net_change_in_correct=net, p_exact_mcnemar=p)


def _verdict(lam, *, valid=True, robust=True, sign=0):
    return SeverityVerdict(lam=lam, diagnostically_valid=valid, robust=robust,
                           sign=sign, n_valid_seeds=4, n_concordant_seeds=4)


def test_T12a_all_valid_severities_retained():
    v = [_verdict(0.25, sign=-1), _verdict(0.50, sign=-1), _verdict(1.00, sign=-1)]
    out = classify_convention(v)
    assert out["valid_severities"] == [0.25, 0.50, 1.00]
    assert out["outcome"] == OUTCOME_A


def test_T12b_robust_opposite_signs_force_F_and_veto_ABD():
    v = [_verdict(0.25, sign=-1), _verdict(0.50, sign=+1), _verdict(1.00, sign=-1)]
    out = classify_convention(v)
    assert out["outcome"] == OUTCOME_F
    assert out["veto_applied"] is True
    assert out["outcome"] not in (OUTCOME_A, OUTCOME_B, OUTCOME_D)
    # every valid severity stays in the record; none is dropped or promoted
    assert out["valid_severities"] == [0.25, 0.50, 1.00]


def test_T12c_letters():
    assert classify_convention([_verdict(l, sign=-1) for l in FROZEN_LAMBDAS]
                               )["outcome"] == OUTCOME_A
    assert classify_convention([_verdict(l, sign=+1) for l in FROZEN_LAMBDAS]
                               )["outcome"] == OUTCOME_B
    assert classify_convention([_verdict(l, robust=False, sign=0)
                                for l in FROZEN_LAMBDAS])["outcome"] == OUTCOME_D
    # a non-robust opposite sign does not trigger the veto
    v = [_verdict(0.25, sign=-1), _verdict(0.50, robust=False, sign=0)]
    assert classify_convention(v)["outcome"] == OUTCOME_A
    # invalid severities are excluded from classification but never "promote" another
    v = [_verdict(0.25, valid=False, sign=+1), _verdict(0.50, sign=-1)]
    out = classify_convention(v)
    assert out["outcome"] == OUTCOME_A and out["valid_severities"] == [0.50]
    # no valid severity at all is not a null
    assert classify_convention([_verdict(0.25, valid=False)]
                               )["outcome"] == OUTCOME_UNDETERMINED


def test_T12d_conventions_classify_independently():
    """Updated for CENTRAL's final joint rule: disagreement -> MIXED_DECODING.

    The intent of the original test is unchanged and still asserted: the two
    conventions are classified independently and neither adjudicates the other.
    Only the joint LABEL changed — CENTRAL final decision §2 replaced the previous
    `F_HETEROGENEOUS` joint label with `MIXED_DECODING` for a material difference.
    """
    from gate_x_lesion.outcomes import JOINT_MIXED_DECODING

    fam = {
        "CANONICAL": [_verdict(l, sign=-1) for l in FROZEN_LAMBDAS],
        "FREE_AR": [_verdict(l, sign=+1) for l in FROZEN_LAMBDAS],
    }
    out = classify_route_family(fam)
    assert out["stages"]["CANONICAL"]["outcome"] == OUTCOME_A
    assert out["stages"]["FREE_AR"]["outcome"] == OUTCOME_B
    # neither adjudicated the other, and both stage letters stay visible
    assert out["joint"]["joint_outcome"] == JOINT_MIXED_DECODING
    assert out["joint"]["canonical"] == OUTCOME_A
    assert out["joint"]["free_ar"] == OUTCOME_B


def test_T12e_convention_disagreement_stays_visible():
    """CENTRAL final decision §2: same letter -> CONCORDANT_X; differ -> MIXED_DECODING."""
    from gate_x_lesion.outcomes import JOINT_MIXED_DECODING, concordant

    # any material difference, including one side being heterogeneous
    assert classify_joint(OUTCOME_A, OUTCOME_B)["joint_outcome"] == JOINT_MIXED_DECODING
    assert classify_joint(OUTCOME_A, OUTCOME_D)["joint_outcome"] == JOINT_MIXED_DECODING
    assert classify_joint(OUTCOME_F, OUTCOME_A)["joint_outcome"] == JOINT_MIXED_DECODING
    assert classify_joint(OUTCOME_A, OUTCOME_F)["joint_outcome"] == JOINT_MIXED_DECODING

    # agreement is concordant, using the project's own (more specific) letters
    assert classify_joint(OUTCOME_A, OUTCOME_A)["joint_outcome"] == concordant(OUTCOME_A)
    assert classify_joint(OUTCOME_D, OUTCOME_D)["joint_outcome"] == concordant(OUTCOME_D)
    assert classify_joint(OUTCOME_F, OUTCOME_F)["joint_outcome"] == concordant(OUTCOME_F)

    # a favourable convention never overwrites a conflicting one
    for a, b in ((OUTCOME_A, OUTCOME_F), (OUTCOME_F, OUTCOME_A)):
        r = classify_joint(a, b)
        assert r["canonical"] == a and r["free_ar"] == b


def test_T12f_severity_rollup_requires_an_explicit_robustness_rule():
    """CLOSURE PASS amendment to T12f.

    NOTE: superseded in part. CENTRAL has since frozen the O-4 rule; see
    `tests/test_gate_x_lesion_final_rules.py`. What this test still pins is that
    `evaluate_severity` accepts NO implicit default, so an unfrozen rule cannot
    return by the back door.

    This test previously exercised a robustness rule baked into `evaluate_severity`
    (>= 3/4 seeds significant at alpha=0.05 plus a pooled check).  That rule was the
    execution agent's own proposal, flagged in the previous handoff as open item O-4;
    it was never frozen by CENTRAL.  Keeping it as a silent default made an unfrozen
    criterion look like preregistration, so it was removed.

    What is asserted now is the fail-closed behaviour plus the parts of the roll-up
    that ARE authoritative: diagnostic validity.  The robustness candidates are
    exercised in `tests/test_gate_x_lesion_closure.py::test_O4_*`, explicitly labelled
    as candidates.
    """
    from gate_x_lesion.outcomes import (CANDIDATE_ROBUSTNESS_RULES,
                                        FROZEN_ROBUSTNESS_RULE,
                                        UnfrozenRobustnessError)

    recs = [_rec(0.25, s, net=-40, p=0.001) for s in range(4)]

    # CENTRAL has since frozen the rule; a rule must still be passed EXPLICITLY,
    # so no unfrozen default can ever slip back in.
    assert FROZEN_ROBUSTNESS_RULE is not None
    with pytest.raises(TypeError):
        evaluate_severity(recs, pooled_p=0.0005, pooled_net=-160)
    with pytest.raises(UnfrozenRobustnessError):
        evaluate_severity(recs, robustness_rule=None)

    # With a CANDIDATE rule supplied explicitly, the roll-up works end to end.
    R3 = CANDIDATE_ROBUSTNESS_RULES["R3_unanimous_sign_no_test"]
    v = evaluate_severity(recs, robustness_rule=R3)
    assert v.diagnostically_valid and v.robust and v.sign == -1

    mixed = [_rec(0.25, 0, net=-40, p=0.001), _rec(0.25, 1, net=-40, p=0.001),
             _rec(0.25, 2, net=+5, p=0.9), _rec(0.25, 3, net=+2, p=0.8)]
    assert evaluate_severity(mixed, robustness_rule=R3).robust is False

    # Diagnostic validity is independent of the robustness rule.
    inert = [_rec(0.25, s, net=0, p=None, changed=0) for s in range(4)]
    assert evaluate_severity(inert, robustness_rule=R3).diagnostically_valid is False

    saturated_acc = [_rec(0.25, s, net=-40, p=0.001, acc=0.001) for s in range(4)]
    assert evaluate_severity(saturated_acc,
                             robustness_rule=R3).diagnostically_valid is False
    saturated_modal = [_rec(0.25, s, net=-40, p=0.001, modal=0.9) for s in range(4)]
    assert evaluate_severity(saturated_modal,
                             robustness_rule=R3).diagnostically_valid is False

    # The authoritative O-3 path can override the local validity heuristic.
    assert evaluate_severity(inert, robustness_rule=R3,
                             diagnostically_valid=True).diagnostically_valid is True


def test_T12g_no_retrospective_promotion_is_structural():
    """The classifier has no interface by which one severity could be selected."""
    import inspect
    src = inspect.getsource(classify_convention)
    for banned in ("max(", "min(", "sorted(valid, key", "[-1]", "[0]"):
        assert banned not in src.replace("sorted(signs)", ""), banned
