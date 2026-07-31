"""Tests for the additive behavioral-evaluation instrumentation patch.

Spec: outputs/behavioral_wfe_fulllexicon_93a577f/audits/
      minimal_behavioral_evaluation_patch_FINAL.md

NO checkpoint is loaded and NO WFE inference is run anywhere in this file.
Every model used here is a tiny randomly initialised fixture or a scripted
stub whose logits are fully controlled by the test.

Groups:
  A  existing-column identity (non-regression)
  B  EOS-position capture
  C  gate / lexical-confidence capture
  D  provenance honesty
  E  manifest join and enrichment
  F  determinism
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pandas as pd
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (Config, DataConfig, WMConfig, LTMConfig, GatingConfig,
                    LossConfig, TrainConfig)
from data.phonemes import build_vocab
from models.dual_route import DualRouteModel
from evaluate.hooks import make_batch
from scripts import external_eval as ee
from scripts import enrich_behavioral_predictions as enr
from scripts.enrich_behavioral_predictions import (
    edit_operations, enrich, frequency_class, _editops_internal,
    resolve_editops_backend, FaithfulBackendUnavailable,
    MODE_FAITHFUL, MODE_ADAPTED, FAITHFUL_BACKEND, ADAPTED_BACKEND,
    REQUIRED_BACKEND_SPEC)

# Frozen cohort gate parameters — deliberately different from the dataclass
# defaults (alpha=4.0, gate_threshold=0.5) so a defaults regression is visible.
FROZEN_ALPHA = 2.0
FROZEN_THRESHOLD = 0.7


# ============================================================  fixtures  ====

def _tiny_config() -> Config:
    return Config(
        data=DataConfig(use_real=False, glove_path=None, semantic_dim=8,
                        max_words=64, val_fraction=0.0,
                        split_mode="full_lexicon", seed=0),
        wm=WMConfig(hidden=16, interference_noise=0.0),
        ltm=LTMConfig(phon_embed_dim=8, enc_hidden=16, dec_hidden=16,
                      ltm_encoder_mode="unigru_last_hidden", ventral_noise=0.0),
        gating=GatingConfig(alpha=FROZEN_ALPHA, usage_prior=0.5,
                            gate_threshold=FROZEN_THRESHOLD),
        loss=LossConfig(), train=TrainConfig(device="cpu", seed=0),
    )


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(20260731)
    vocab = build_vocab()
    cfg = _tiny_config()
    model = DualRouteModel(cfg, vocab)
    bank = torch.randn(32, cfg.data.semantic_dim)
    model.set_semantic_bank(bank)
    model.eval()
    return model, vocab, cfg


@pytest.fixture(scope="module")
def tiny_forms(tiny_model):
    _, vocab, _ = tiny_model
    p = [vocab.stoi[s] for s in ("K", "AE", "T", "S", "IH", "NG", "B", "AH")]
    return [p[:3], p[:5], p[:8], p[:4]]     # unequal lengths, 4 items


class ScriptedModel:
    """Stub exposing only `route_logits`, emitting a fixed token per step.

    `script[route]` is a list of per-step token ids, one entry per batch item:
    script[route][step][item].  Steps beyond the script repeat its last row.
    """

    def __init__(self, script, vocab_size: int):
        self.script = script
        self.vocab_size = vocab_size

    def route_logits(self, enc_in, enc_mask, dec_in, route: str,
                     collect: bool = False, apply_noise: bool = False):
        step = dec_in.shape[1] - 1          # 0-based generation step
        rows = self.script[route]
        toks = rows[min(step, len(rows) - 1)]
        B, S = dec_in.shape
        logits = torch.zeros(B, S, self.vocab_size)
        for i, t in enumerate(toks):
            logits[i, -1, t] = 10.0
        return {"logits": logits}


# ======================================================  A  non-regression ==

def _prepatch_autoregressive_decode_batch(model, vocab, forms, device,
                                          routes=("full", "wm", "ltm"),
                                          wm_noise=False):
    """Verbatim pre-patch implementation (HEAD aacb653) as golden reference."""
    batch = make_batch(forms, vocab, device)
    max_steps = max(len(f) for f in forms)
    preds_by_route = {}
    for route in routes:
        collect = (route == "wm") and wm_noise
        dec_input = batch["enc_in"].new_full((len(forms), 1), vocab.bos_id)
        for _ in range(max_steps):
            res = model.route_logits(batch["enc_in"], batch["enc_mask"],
                                     dec_input, route=route, collect=collect)
            next_tok = res["logits"][:, -1, :].argmax(-1, keepdim=True)
            dec_input = torch.cat([dec_input, next_tok], dim=1)
        route_preds = []
        for i, form in enumerate(forms):
            n_steps = len(form)
            pred_ids = dec_input[i, 1:n_steps + 1].tolist()
            seq = []
            for idx in pred_ids:
                if idx == vocab.eos_id:
                    break
                seq.append(idx)
            route_preds.append(seq)
        preds_by_route[route] = route_preds
    return preds_by_route


def test_A_ar_predictions_identical_to_prepatch(tiny_model, tiny_forms):
    model, vocab, _ = tiny_model
    golden = _prepatch_autoregressive_decode_batch(model, vocab, tiny_forms, "cpu")
    plain = ee.autoregressive_decode_batch(model, vocab, tiny_forms, "cpu")
    instrumented, _ = ee.autoregressive_decode_batch(
        model, vocab, tiny_forms, "cpu", return_instrumentation=True)
    assert plain == golden, "default return value drifted from pre-patch"
    assert instrumented == golden, "instrumentation altered the predictions"


def test_A_default_return_type_unchanged(tiny_model, tiny_forms):
    model, vocab, _ = tiny_model
    out = ee.autoregressive_decode_batch(model, vocab, tiny_forms, "cpu")
    assert isinstance(out, dict) and not isinstance(out, tuple)
    out_tf = ee.eval_batch(model, vocab, tiny_forms, "cpu")
    assert isinstance(out_tf, dict) and not isinstance(out_tf, tuple)
    res = ee.evaluate_items(model, vocab, tiny_forms, "cpu", batch_size=2,
                            decode=ee.DECODE_AR)
    assert isinstance(res, dict) and set(res) == {"full", "wm", "ltm"}


def test_A_existing_metric_columns_byte_identical(tiny_model, tiny_forms, tmp_path):
    """Pre-patch table must be the patched table minus the appended columns."""
    model, vocab, _ = tiny_model
    routes = ("full", "wm", "ltm")
    base = pd.DataFrame({"item_id": [f"toy_{i:04d}" for i in range(len(tiny_forms))],
                         "lexicality": ["real", "pseudo", "real", "pseudo"],
                         "zipf_frequency": [4.5, None, 3.1, None]})

    res_plain = ee.evaluate_items(model, vocab, tiny_forms, "cpu", batch_size=2,
                                  routes=routes, decode=ee.DECODE_AR)
    df_pre = base.copy()
    for r in routes:
        for m in ("exact_match", "phoneme_acc", "edit_dist", "norm_edit",
                  "predicted", "target"):
            df_pre[f"{r}_{m}"] = [x[m] for x in res_plain[r]]

    res, instr = ee.evaluate_items(model, vocab, tiny_forms, "cpu", batch_size=2,
                                   routes=routes, decode=ee.DECODE_AR,
                                   return_instrumentation=True)
    df_post = base.copy()
    for r in routes:
        for m in ("exact_match", "phoneme_acc", "edit_dist", "norm_edit",
                  "predicted", "target"):
            df_post[f"{r}_{m}"] = [x[m] for x in res[r]]
    df_post = ee.attach_instrumentation_columns(df_post, instr, routes)

    # Header: pre-patch columns are an exact ordered prefix of the patched ones
    assert list(df_post.columns)[:len(df_pre.columns)] == list(df_pre.columns)
    new_cols = list(df_post.columns)[len(df_pre.columns):]
    assert new_cols == [f"{r}_eos_position" for r in routes] + [
        "gate", "lexical_confidence", "lexical_margin", "lexical_density"]

    # Bytes: serialising the patched table with the new columns dropped
    # reproduces the pre-patch file exactly.
    pre = tmp_path / "pre.tsv"
    post_trimmed = tmp_path / "post_trimmed.tsv"
    df_pre.to_csv(pre, sep="\t", index=False)
    df_post.drop(columns=new_cols).to_csv(post_trimmed, sep="\t", index=False)
    assert pre.read_bytes() == post_trimmed.read_bytes()

    # Row order and item order preserved
    assert list(df_post["item_id"]) == list(base["item_id"])


def test_A_teacher_forced_path_unchanged(tiny_model, tiny_forms):
    model, vocab, _ = tiny_model
    plain = ee.eval_batch(model, vocab, tiny_forms, "cpu")
    instrumented, instr = ee.eval_batch(model, vocab, tiny_forms, "cpu",
                                        return_instrumentation=True)
    assert plain == instrumented
    # TF has no AR generation window: eos_position is absent by construction
    assert all(v is None for r in instr["eos_position"]
               for v in instr["eos_position"][r])


# ==========================================================  B  EOS position =

def test_B_eos_first_middle_absent_and_after_window(tiny_model):
    _, vocab, _ = tiny_model
    eos, pad = vocab.eos_id, vocab.pad_id
    K, AE, T = vocab.stoi["K"], vocab.stoi["AE"], vocab.stoi["T"]
    forms = [[K, AE, T], [K, AE, T], [K, AE, T], [K, AE]]
    #        item0: EOS first   item1: EOS middle  item2: no EOS
    #        item3: length 2, EOS only at step 2 → outside its window
    script = {r: [[eos, K, K, K], [K, eos, AE, K], [K, T, T, eos]]
              for r in ("full", "wm", "ltm")}
    model = ScriptedModel(script, vocab.size)

    preds, instr = ee.autoregressive_decode_batch(
        model, vocab, forms, "cpu", return_instrumentation=True)
    pos = instr["eos_position"]["full"]

    assert pos[0] == 0, "EOS as the first generated token must be position 0"
    assert pos[1] == 1, "EOS at the second position must be position 1"
    assert pos[2] is None, "no EOS in the window must be None"
    assert pos[3] is None, "EOS beyond the item's window must not be recorded"

    # eos_position is an index into the readout window, never a length and
    # never 1-based: len(prediction) == eos_position whenever EOS is present.
    assert len(preds["full"][0]) == 0 and pos[0] == 0
    assert len(preds["full"][1]) == 1 and pos[1] == 1
    assert len(preds["full"][2]) == 3 and pos[2] is None
    assert pad not in preds["full"][2]


def test_B_tokens_after_eos_do_not_move_the_position(tiny_model):
    _, vocab, _ = tiny_model
    eos = vocab.eos_id
    K, AE, T = vocab.stoi["K"], vocab.stoi["AE"], vocab.stoi["T"]
    forms = [[K, AE, T, K, AE]]
    script = {r: [[K], [eos], [T], [eos], [AE]] for r in ("full", "wm", "ltm")}
    model = ScriptedModel(script, vocab.size)
    preds, instr = ee.autoregressive_decode_batch(
        model, vocab, forms, "cpu", return_instrumentation=True)
    assert instr["eos_position"]["full"][0] == 1        # FIRST eos only
    assert preds["full"][0] == [K]


def test_B_unequal_lengths_and_final_partial_batch(tiny_model, tiny_forms):
    model, vocab, _ = tiny_model
    routes = ("full", "wm", "ltm")
    # batch_size=3 over 4 items → a final partial batch of 1
    res, instr = ee.evaluate_items(model, vocab, tiny_forms, "cpu", batch_size=3,
                                   routes=routes, decode=ee.DECODE_AR,
                                   return_instrumentation=True)
    for r in routes:
        assert len(instr["eos_position"][r]) == len(tiny_forms)
        for i, v in enumerate(instr["eos_position"][r]):
            assert v is None or 0 <= v < len(tiny_forms[i])
            if v is not None:
                assert len(res[r][i]["predicted"].split()) == v


def test_B_position_helper_conventions(tiny_model):
    _, vocab, _ = tiny_model
    eos = vocab.eos_id
    assert ee._first_eos_position([eos, 5, 6], eos) == 0
    assert ee._first_eos_position([5, eos, 6], eos) == 1
    assert ee._first_eos_position([5, 6, 7], eos) is None
    assert ee._first_eos_position([], eos) is None


def test_B_missing_value_is_empty_not_zero(tiny_model, tiny_forms):
    model, vocab, _ = tiny_model
    routes = ("full",)
    _, instr = ee.evaluate_items(model, vocab, tiny_forms, "cpu", batch_size=4,
                                 routes=routes, decode=ee.DECODE_AR,
                                 return_instrumentation=True)
    instr["eos_position"]["full"] = [None] * len(tiny_forms)
    df = pd.DataFrame({"item_id": range(len(tiny_forms))})
    df = ee.attach_instrumentation_columns(df, instr, routes)
    assert list(df["full_eos_position"]) == [ee.MISSING] * len(tiny_forms)
    assert ee.MISSING == "" and ee.MISSING != 0


# ====================================================  C  gate / confidence ==

def test_C_gate_matches_model_internal_value(tiny_model, tiny_forms):
    model, vocab, cfg = tiny_model
    batch = make_batch(tiny_forms, vocab, "cpu")
    captured = ee.capture_gate_and_field(model, batch, vocab)

    with torch.no_grad():
        ref = model.forward(batch["enc_in"], batch["enc_mask"], batch["dec_in"])
    ref_gate = ref["gate"][:, 0, 0].tolist()
    ref_conf = ref["field_confidence"].tolist()

    assert captured["gate"] == pytest.approx(ref_gate, abs=0.0)
    assert captured["lexical_confidence"] == pytest.approx(ref_conf, abs=0.0)
    # gate is word-level: identical at every decoder timestep
    assert torch.allclose(ref["gate"][:, 0, 0], ref["gate"][:, -1, 0])


def test_C_gate_uses_checkpoint_parameters_not_defaults(tiny_model, tiny_forms):
    model, vocab, cfg = tiny_model
    assert cfg.gating.alpha == FROZEN_ALPHA
    assert cfg.gating.gate_threshold == FROZEN_THRESHOLD
    assert (GatingConfig().alpha, GatingConfig().gate_threshold) == (4.0, 0.5)

    batch = make_batch(tiny_forms, vocab, "cpu")
    cap = ee.capture_gate_and_field(model, batch, vocab)
    conf = torch.tensor(cap["lexical_confidence"])
    expected = torch.sigmoid(FROZEN_ALPHA * (conf - FROZEN_THRESHOLD))
    assert torch.allclose(torch.tensor(cap["gate"]), expected, atol=1e-6)
    wrong = torch.sigmoid(4.0 * (conf - 0.5))
    assert not torch.allclose(torch.tensor(cap["gate"]), wrong, atol=1e-6)


def test_C_confidence_is_top1_similarity_not_index_margin_or_density(tiny_model,
                                                                     tiny_forms):
    model, vocab, _ = tiny_model
    batch = make_batch(tiny_forms, vocab, "cpu")
    cap = ee.capture_gate_and_field(model, batch, vocab)
    with torch.no_grad():
        s_hat = model.ltm.encode(batch["enc_in"], batch["enc_mask"])
        field = model.ltm.lexical_field(s_hat)
    assert cap["lexical_confidence"] == pytest.approx(
        field["confidence"].tolist(), abs=0.0)
    assert cap["lexical_margin"] == pytest.approx(field["margin"].tolist(), abs=0.0)
    assert cap["lexical_density"] == pytest.approx(field["density"].tolist(), abs=0.0)
    # confidence is a similarity in [-1, 1], never a bank index
    assert all(-1.0001 <= c <= 1.0001 for c in cap["lexical_confidence"])
    assert cap["lexical_confidence"] != cap["lexical_margin"]


def test_C_capture_does_not_alter_predictions_or_bank(tiny_model, tiny_forms):
    model, vocab, _ = tiny_model
    bank_before = model.ltm.semantic_bank.clone()
    before = ee.autoregressive_decode_batch(model, vocab, tiny_forms, "cpu")
    batch = make_batch(tiny_forms, vocab, "cpu")
    ee.capture_gate_and_field(model, batch, vocab)
    after = ee.autoregressive_decode_batch(model, vocab, tiny_forms, "cpu")
    assert before == after
    assert torch.equal(bank_before, model.ltm.semantic_bank)
    assert not model.training


def test_C_wm_and_ltm_carry_no_fabricated_gate(tiny_model, tiny_forms):
    """Gate/confidence are item-level FULL-route columns; no per-route gate."""
    model, vocab, _ = tiny_model
    routes = ("full", "wm", "ltm")
    res, instr = ee.evaluate_items(model, vocab, tiny_forms, "cpu", batch_size=4,
                                   routes=routes, decode=ee.DECODE_AR,
                                   return_instrumentation=True)
    df = pd.DataFrame({"item_id": range(len(tiny_forms))})
    df = ee.attach_instrumentation_columns(df, instr, routes)
    for r in ("wm", "ltm"):
        assert f"{r}_gate" not in df.columns
        assert f"{r}_lexical_confidence" not in df.columns
    assert "gate" in df.columns and "lexical_confidence" in df.columns
    assert all(isinstance(v, float) for v in df["gate"])


def test_C_full_precision_not_rounded(tiny_model, tiny_forms):
    model, vocab, _ = tiny_model
    batch = make_batch(tiny_forms, vocab, "cpu")
    cap = ee.capture_gate_and_field(model, batch, vocab)
    assert any(len(repr(g).split(".")[-1]) > 6 for g in cap["gate"]), \
        "gate values look rounded for display"


def test_C_no_bank_yields_missing_not_fabricated(tiny_forms):
    torch.manual_seed(7)
    vocab = build_vocab()
    model = DualRouteModel(_tiny_config(), vocab)   # bank left at size 1
    model.eval()
    batch = make_batch(tiny_forms, vocab, "cpu")
    cap = ee.capture_gate_and_field(model, batch, vocab)
    assert all(c is None for c in cap["lexical_confidence"])
    df = ee.attach_instrumentation_columns(
        pd.DataFrame({"item_id": range(len(tiny_forms))}),
        {"eos_position": {"full": [None] * len(tiny_forms)}, **cap}, ("full",))
    assert list(df["lexical_confidence"]) == [ee.MISSING] * len(tiny_forms)


# ============================================================  D  provenance =

def _fake_meta(tmp_path) -> dict:
    ckpt = tmp_path / "fake_ckpt.pt"
    ckpt.write_bytes(b"not-a-real-checkpoint")
    return {"_provenance_inputs": {
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "checkpoint_training_commit": "93a577fd9822955fa272ee733fa7e2acf81f1333",
        "checkpoint_epoch": 140, "seed": 22, "glove_path": None,
        "gate_alpha": FROZEN_ALPHA, "gate_threshold": FROZEN_THRESHOLD,
    }}


def test_D_required_fields_present(tmp_path):
    wfe = tmp_path / "wfe.tsv"
    wfe.write_text("item_id\tword\nwfe_0000\tcat\n")
    prov = ee.build_provenance(_fake_meta(tmp_path), wfe_tsv=str(wfe),
                               ssp_tsv=None, manifest_path=None, device="cpu",
                               decode=ee.DECODE_AR, routes=("full", "wm", "ltm"),
                               wm_noise=False, argv=["external_eval.py", "--x"])
    for field in ("checkpoint_training_commit", "evaluation_code_base_commit",
                  "evaluation_code_dirty", "evaluation_patch_diff_sha256",
                  "seed", "checkpoint_epoch", "checkpoint_sha256",
                  "wfe_tsv_path", "wfe_tsv_sha256", "device", "decode_mode",
                  "routes", "deterministic_no_noise", "decode_convention",
                  "eos_position_convention", "gate_convention",
                  "python_version", "torch_version", "command",
                  "timestamp_utc", "output_schema_version"):
        assert field in prov, f"missing provenance field {field}"
    assert prov["decode_convention"] == ee.DECODE_CONVENTION_NOTE
    assert prov["deterministic_no_noise"] is True
    assert prov["apply_noise"] is False
    assert prov["teacher_forcing_in_primary_path"] is False
    json.dumps(prov)        # must be serialisable


def test_D_hashes_refer_to_the_files_actually_used(tmp_path):
    wfe = tmp_path / "wfe.tsv"
    wfe.write_text("item_id\nwfe_0000\n")
    prov = ee.build_provenance(_fake_meta(tmp_path), wfe_tsv=str(wfe),
                               ssp_tsv=None, manifest_path=None, device="cpu",
                               decode=ee.DECODE_AR, routes=("full",),
                               wm_noise=False)
    assert prov["wfe_tsv_sha256"] == hashlib.sha256(wfe.read_bytes()).hexdigest()
    wfe.write_text("item_id\nwfe_0001\n")          # change the file
    prov2 = ee.build_provenance(_fake_meta(tmp_path), wfe_tsv=str(wfe),
                                ssp_tsv=None, manifest_path=None, device="cpu",
                                decode=ee.DECODE_AR, routes=("full",),
                                wm_noise=False)
    assert prov2["wfe_tsv_sha256"] != prov["wfe_tsv_sha256"]


def test_D_training_commit_distinct_from_evaluation_code(tmp_path):
    prov = ee.build_provenance(_fake_meta(tmp_path), wfe_tsv=None, ssp_tsv=None,
                               manifest_path=None, device="cpu",
                               decode=ee.DECODE_AR, routes=("full",),
                               wm_noise=False)
    assert prov["checkpoint_training_commit"].startswith("93a577fd")
    assert prov["evaluation_code_base_commit"] != prov["checkpoint_training_commit"]


def test_D_source_hashes_cover_untracked_new_files(tmp_path):
    """Code identity must not depend on files being tracked by git."""
    prov = ee.build_provenance(_fake_meta(tmp_path), wfe_tsv=None, ssp_tsv=None,
                               manifest_path=None, device="cpu",
                               decode=ee.DECODE_AR, routes=("full",),
                               wm_noise=False)
    src = prov["evaluation_source_sha256"]
    for rel in ("scripts/external_eval.py",
                "scripts/enrich_behavioral_predictions.py",
                "models/gating.py", "models/dual_route.py"):
        assert src.get(rel), f"no source hash recorded for {rel}"
        on_disk = hashlib.sha256(
            open(os.path.join(ROOT, rel), "rb").read()).hexdigest()
        assert src[rel] == on_disk


def test_D_dirty_tree_never_fabricates_a_commit(tmp_path):
    prov = ee.build_provenance(_fake_meta(tmp_path), wfe_tsv=None, ssp_tsv=None,
                               manifest_path=None, device="cpu",
                               decode=ee.DECODE_AR, routes=("full",),
                               wm_noise=False)
    real_dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT).decode().strip())
    assert prov["evaluation_code_dirty"] == real_dirty
    if prov["evaluation_code_dirty"] is not False:
        assert prov["evaluation_code_commit"] is None, \
            "a dirty run must not claim a final evaluation_code_commit"
    assert len(prov["evaluation_patch_diff_sha256"]) in (64, len("unknown"))


# ==================================================  E  manifest enrichment ==

MANIFEST = os.path.join(ROOT, "outputs", "behavioral_wfe_fulllexicon_93a577f",
                        "audits", "wfe_analysis_item_manifest.tsv")
SETS = os.path.join(ROOT, "outputs", "behavioral_wfe_fulllexicon_93a577f",
                    "audits", "wfe_analysis_set_membership.tsv")
requires_manifest = pytest.mark.skipif(
    not os.path.exists(MANIFEST), reason="frozen manifest not present")


def _toy_predictions(man: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    sub = man.head(n)
    rows = []
    for _, m in sub.iterrows():
        tgt = m["target_phonemes"]
        toks = str(tgt).split()
        pred = " ".join(toks[:-1]) if len(toks) > 3 else tgt   # some deletions
        rows.append({"item_id": m["item_id"], "word": m["word"],
                     "lexicality": m["source_lexicality"],
                     "condition": m["condition"],
                     "zipf_frequency": m["zipf_frequency"],
                     "length_phonemes": m["phoneme_length"]})
        for r in ("full", "wm", "ltm"):
            rows[-1][f"{r}_target"] = tgt
            rows[-1][f"{r}_predicted"] = pred
            rows[-1][f"{r}_exact_match"] = int(pred == tgt)
            ops = edit_operations(str(tgt).split(), str(pred).split())
            rows[-1][f"{r}_edit_dist"] = ops["edit_distance"]
    return pd.DataFrame(rows)


@requires_manifest
def test_E_join_is_one_to_one_and_complete():
    man = pd.read_csv(MANIFEST, sep="\t")
    sets = pd.read_csv(SETS, sep="\t")
    assert len(man) == 1200 and man["item_id"].is_unique
    pred = _toy_predictions(man)
    out, report = enrich(pred, man, sets)
    assert len(out) == len(pred)
    assert out["item_id"].is_unique
    assert list(out["item_id"]) == list(pred["item_id"])
    assert report["join_is_one_to_one"] and report["n_unmatched_predictions"] == 0


@requires_manifest
def test_E_unmatched_item_raises_instead_of_dropping():
    man = pd.read_csv(MANIFEST, sep="\t")
    pred = _toy_predictions(man)
    pred.loc[0, "item_id"] = "wfe_9999"
    with pytest.raises(AssertionError, match="absent from the manifest"):
        enrich(pred, man, None)


@requires_manifest
def test_E_duplicate_item_id_raises():
    man = pd.read_csv(MANIFEST, sep="\t")
    pred = _toy_predictions(man)
    pred.loc[1, "item_id"] = pred.loc[0, "item_id"]
    with pytest.raises(AssertionError, match="duplicate item_id"):
        enrich(pred, man, None)


@requires_manifest
def test_E_exposure_and_sets_copied_exactly():
    man = pd.read_csv(MANIFEST, sep="\t")
    sets = pd.read_csv(SETS, sep="\t")
    pred = _toy_predictions(man)
    out, report = enrich(pred, man, sets)
    man_idx = man.set_index("item_id")
    for item in out["item_id"]:
        row = out[out["item_id"] == item].iloc[0]
        assert row["lichtheim_exposure_status"] == \
            man_idx.at[item, "lichtheim_exposure_status"]
        assert row["dager_collision_status"] == \
            man_idx.at[item, "dager_collision_status"]
    truth = set(sets.loc[sets["analysis_set"] == "LICHTHEIM_CLEAN", "item_id"])
    assert list(out["in_LICHTHEIM_CLEAN"]) == [i in truth for i in out["item_id"]]


@requires_manifest
def test_E_no_serial_position_field_is_derived():
    man = pd.read_csv(MANIFEST, sep="\t")
    pred = _toy_predictions(man)
    out, report = enrich(pred, man, None)
    banned = [c for c in out.columns
              if "error_indices" in c.lower() or "serial_position" in c.lower()
              or "relative_position" in c.lower()]
    assert banned == [], f"aligned-op serial-position leak: {banned}"
    assert report["serial_position_fields_derived"] is False


def test_E_operations_sum_to_edit_distance():
    cases = [(["K", "AE", "T"], ["K", "AE", "T"]),
             (["K", "AE", "T"], ["K", "EH", "T"]),
             (["S", "EH", "K"], ["EH", "K"]),
             (["K", "AE", "T"], ["K", "AE", "T", "S"]),
             (["K", "AE", "T"], []),
             ([], ["K"])]
    for tgt, pred in cases:
        ops = edit_operations(tgt, pred)
        assert (ops["insertions"] + ops["deletions"] + ops["substitutions"]
                == ops["edit_distance"])
        assert ops["edit_distance"] == len(_editops_internal(tgt, pred))


def test_E_operation_direction_matches_dager():
    """editops(gold -> prediction): a shorter prediction is a DELETION."""
    ops = edit_operations(["S", "EH", "K"], ["EH", "K"])
    assert ops["deletions"] == 1 and ops["insertions"] == 0
    ops = edit_operations(["EH", "K"], ["S", "EH", "K"])
    assert ops["insertions"] == 1 and ops["deletions"] == 0


def test_E_frequency_class_thresholds():
    assert frequency_class("real", 3.5) == "low"
    assert frequency_class("real", 4.0) == "high"
    assert frequency_class("real", 3.75) == "ambiguous_3.5_4.0"
    assert frequency_class("real", None) == "missing_zipf"
    assert frequency_class("pseudo", None) == "n/a_pseudoword"


@requires_manifest
def test_E_premature_eos_and_predicted_length():
    man = pd.read_csv(MANIFEST, sep="\t")
    pred = _toy_predictions(man)
    out, _ = enrich(pred, man, None)
    for _, row in out.iterrows():
        n_pred = len(str(row["full_predicted"]).split())
        n_tgt = len(str(row["full_target"]).split())
        assert row["full_predicted_length"] == n_pred
        assert bool(row["full_premature_eos"]) == (n_pred < n_tgt)


# ==============================================  G  editops backend policy ==
#
# Dager evidence (danieldager/swp-model @ dc09eb3):
#   requirements.txt:13         levenshtein>=0.26.1
#   swp/utils/datasets.py:11    from Levenshtein import editops
#   scripts/jeanzay_setup.sh:18 pip install --upgrade --no-cache-dir levenshtein
#
# What the backend choice does and does not change:
#   NOT Figures 2A/2B — they use only the TOTAL raw edit distance, identical
#                       across all optimal alignments;
#   NOT Figure 2C     — zip-mismatch Error_Indices, no alignment at all;
#   IT DOES affect    — the insertion/deletion/substitution taxonomy, Figure
#                       8-style error-type analyses, and any separately
#                       labelled aligned-error extension.

def test_G_faithful_mode_uses_dager_backend_and_records_version():
    backend = resolve_editops_backend(MODE_FAITHFUL)
    assert backend["editops_backend"] == FAITHFUL_BACKEND == "Levenshtein.editops"
    assert backend["editops_backend_package"] == "Levenshtein"
    assert backend["publication_eligible"] is True
    version = backend["editops_backend_version"]
    assert version, "faithful backend must record a package version"
    major, minor = (int(x) for x in version.split(".")[:2])
    assert (major, minor) >= (0, 26), \
        f"installed Levenshtein {version} violates {REQUIRED_BACKEND_SPEC}"


def test_G_faithful_mode_fails_when_backend_unavailable(monkeypatch):
    """No silent fallback: faithful mode must fail loudly without Levenshtein."""
    monkeypatch.setattr(enr, "_lev", None)
    with pytest.raises(FaithfulBackendUnavailable) as exc:
        resolve_editops_backend(MODE_FAITHFUL)
    msg = str(exc.value)
    assert REQUIRED_BACKEND_SPEC in msg and "pip install" in msg
    # the same failure must propagate through the public entry points
    with pytest.raises(FaithfulBackendUnavailable):
        edit_operations(["K", "AE", "T"], ["K", "AE"], mode=MODE_FAITHFUL)
    with pytest.raises(FaithfulBackendUnavailable):
        enrich(pd.DataFrame({"item_id": ["x"]}),
               pd.DataFrame({"item_id": ["x"]}), None, mode=MODE_FAITHFUL)


def test_G_adapted_mode_is_explicitly_labelled(monkeypatch):
    backend = resolve_editops_backend(MODE_ADAPTED)
    assert backend["editops_backend"] == ADAPTED_BACKEND
    assert backend["publication_eligible"] is False
    assert backend["ADAPTED_NON_PUBLICATION"] is True
    assert "NOT the Dager backend" in backend["editops_tie_breaking"]
    assert "must not be used" in backend["warning"]
    # adapted mode still works with the library absent — that is its purpose
    monkeypatch.setattr(enr, "_lev", None)
    assert resolve_editops_backend(MODE_ADAPTED)["publication_eligible"] is False
    ops = edit_operations(["S", "EH", "K"], ["EH", "K"], mode=MODE_ADAPTED)
    assert ops["edit_distance"] == 1


@requires_manifest
def test_G_mode_is_recorded_on_every_row_and_in_the_report():
    man = pd.read_csv(MANIFEST, sep="\t")
    pred = _toy_predictions(man)
    for mode, eligible in ((MODE_FAITHFUL, True), (MODE_ADAPTED, False)):
        out, report = enrich(pred, man, None, mode=mode)
        assert set(out["analysis_mode"]) == {mode}
        assert report["analysis_mode"] == mode
        assert report["publication_eligible"] is eligible
        assert set(out["editops_backend"]) == {report["editops_backend"]}
        assert report["dager_backend_evidence"]["requirements.txt:13"] == \
            "levenshtein>=0.26.1"
    out_f, _ = enrich(pred, man, None, mode=MODE_FAITHFUL)
    assert set(out_f["editops_backend_version"]) != {""}


def test_G_backends_agree_on_the_total_even_when_split_may_differ():
    cases = [(["K", "AE", "T"], ["K", "EH", "T"]),
             (["S", "EH", "K"], ["EH", "K"]),
             (["K", "AE", "T"], ["K", "AE", "T", "S"]),
             (["AH", "T", "EH", "N", "D", "IH", "NG"], ["AH", "T", "EH", "N"]),
             (["K", "AE", "T"], ["T", "AE", "K"])]
    for tgt, pred in cases:
        f = edit_operations(tgt, pred, mode=MODE_FAITHFUL)
        a = edit_operations(tgt, pred, mode=MODE_ADAPTED)
        assert f["edit_distance"] == a["edit_distance"], (tgt, pred)
        for d in (f, a):
            assert d["insertions"] + d["deletions"] + d["substitutions"] \
                == d["edit_distance"]


def test_G_default_mode_is_faithful():
    import inspect
    assert inspect.signature(edit_operations).parameters["mode"].default \
        == MODE_FAITHFUL
    assert inspect.signature(enrich).parameters["mode"].default == MODE_FAITHFUL


def test_G_evaluation_provenance_records_the_backend(tmp_path):
    prov = ee.build_provenance(_fake_meta(tmp_path), wfe_tsv=None, ssp_tsv=None,
                               manifest_path=None, device="cpu",
                               decode=ee.DECODE_AR, routes=("full",),
                               wm_noise=False)
    assert prov["levenshtein_required_spec"] == "Levenshtein>=0.26.1"
    assert prov["levenshtein_package_version"], \
        "evaluation provenance must record the installed Levenshtein version"
    assert "editops_backend_for_operation_counts" in prov


def test_G_requirements_declares_the_backend():
    req = open(os.path.join(ROOT, "requirements.txt")).read()
    assert "Levenshtein>=0.26.1" in req
    assert "python-Levenshtein" not in req.replace(
        "NOT the legacy `python-Levenshtein`", "")


# ===========================================================  F  determinism =

def test_F_instrumentation_is_deterministic(tiny_model, tiny_forms, tmp_path):
    model, vocab, _ = tiny_model
    routes = ("full", "wm", "ltm")
    paths = []
    for run in range(2):
        res, instr = ee.evaluate_items(model, vocab, tiny_forms, "cpu",
                                       batch_size=2, routes=routes,
                                       decode=ee.DECODE_AR,
                                       return_instrumentation=True)
        df = pd.DataFrame({"item_id": [f"toy_{i}" for i in range(len(tiny_forms))]})
        for r in routes:
            for m in ("exact_match", "edit_dist", "predicted", "target"):
                df[f"{r}_{m}"] = [x[m] for x in res[r]]
        df = ee.attach_instrumentation_columns(df, instr, routes)
        p = tmp_path / f"run{run}.tsv"
        df.to_csv(p, sep="\t", index=False)
        paths.append(p)
    assert paths[0].read_bytes() == paths[1].read_bytes()


def test_F_no_checkpoint_or_wfe_inference_in_this_module():
    """Guard: these tests must never load a checkpoint or decode the real WFE.

    Inspects the CALLS this module makes (AST), not its raw text, so the guard
    cannot be satisfied or broken by the names appearing inside string
    literals — including the ones in this test.
    """
    import ast

    banned = {"load", "load_model_and_vocab", "run_wfe_eval", "run_ssp_eval",
              "main"}
    tree = ast.parse(open(os.path.abspath(__file__)).read())
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                called.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                called.add(fn.attr)
    assert banned.isdisjoint(called), \
        f"forbidden call(s) in the patch tests: {sorted(banned & called)}"
