"""Tests for the final publication and closure release.

This release is editorial: it must add no scientific value and change no
scientific value.  These tests are therefore mostly non-regression and wording
guards rather than numerical checks of new estimates — there are none.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.behavioral_analysis import common
from scripts.behavioral_analysis import final_release as fr
from scripts.behavioral_analysis import plot_final_release as pfr

PKG_DIR = os.path.join(ROOT, "scripts", "behavioral_analysis")
REPORT = os.path.join(ROOT, "reports", "behavioral_wfe_fulllexicon_93a577f")
Q = os.path.join(REPORT, "final_release")
CTL = os.path.join(Q, "_control")
TAB = os.path.join(Q, "tables")
FIG_M, FIG_S = os.path.join(Q, "figures", "main"), os.path.join(Q, "figures", "supplementary")
CAP_M, CAP_S = os.path.join(Q, "captions", "main"), os.path.join(Q, "captions", "supplementary")
FE = os.path.join(Q, "formatted_existing")
LEGACY = fr.LEGACY_DIR

FR_MODULES = ["final_release.py", "plot_final_release.py"]

requires_outputs = pytest.mark.skipif(
    not os.path.exists(os.path.join(TAB, "final_figure_index.tsv")),
    reason="final release not generated")

LIVING = {"reports/behavioral_wfe_fulllexicon_93a577f/README.md",
          "reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv"}
MANIFESTS = [
    ("production", "outputs/behavioral_wfe_fulllexicon_93a577f/full_wfe_evaluation/"
                   "_control/production_scientific_outputs_FINAL.sha256", frozenset()),
    ("sprint1", "reports/behavioral_wfe_fulllexicon_93a577f/validation/"
                "sprint1_outputs.sha256", LIVING),
    ("morphology", "reports/behavioral_wfe_fulllexicon_93a577f/morphology/"
                   "validation/morphology_outputs.sha256", LIVING),
    ("frequency", "reports/behavioral_wfe_fulllexicon_93a577f/frequency/"
                  "validation/frequency_outputs.sha256", LIVING),
    ("error_taxonomy", "reports/behavioral_wfe_fulllexicon_93a577f/error_taxonomy/"
                       "validation/error_taxonomy_outputs.sha256", frozenset()),
    ("feature_importance", "reports/behavioral_wfe_fulllexicon_93a577f/"
                           "feature_importance/validation/"
                           "feature_importance_outputs.sha256", frozenset()),
]


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def _read(p):
    return pd.read_csv(p, sep="\t")


def _text(*paths):
    return "\n".join(open(p).read() for p in paths)


def _norm(s):
    s = s.replace("`", "").replace("*", "")
    s = re.sub(r"(?m)^\s*>\s?", "", s)
    return " ".join(s.split())


def _code_only(module):
    """Source with docstrings and comment-only lines removed.

    The modules document the rules they follow ("no checkpoint", the path of the
    faithful outputs), so a raw substring scan would flag the prose stating a
    rule as a breach of it.
    """
    src = open(os.path.join(PKG_DIR, module)).read()
    lines, drop = src.split("\n"), set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        b = getattr(node, "body", None)
        if (b and isinstance(b[0], ast.Expr)
                and isinstance(b[0].value, ast.Constant)
                and isinstance(b[0].value.value, str)):
            d = b[0].value
            drop.update(range(d.lineno - 1, (d.end_lineno or d.lineno)))
    return "\n".join(l for i, l in enumerate(lines)
                     if i not in drop and not l.lstrip().startswith("#"))


def _all_release_docs():
    out = []
    for base, _, names in os.walk(Q):
        out += [os.path.join(base, n) for n in sorted(names) if n.endswith(".md")]
    return out


def _verify(manifest, allow=frozenset()):
    bad = []
    with open(manifest) as f:
        for line in f:
            h, _, rel = line.strip().partition("  ")
            if not rel or rel in allow:
                continue
            p = os.path.join(ROOT, rel)
            if not os.path.exists(p) or _sha(p) != h:
                bad.append(rel)
    return bad


# ==============================  A  repository and source integrity =========

def test_A_canonical_table_unchanged():
    prov = json.load(open(os.path.join(
        REPORT, "behavioral_analysis_provenance.json")))
    assert _sha(common.CANONICAL_TABLE) == prov["canonical_table_sha256"]


@pytest.mark.parametrize("label,rel,allow", MANIFESTS)
def test_A_prior_manifests_verify(label, rel, allow):
    m = os.path.join(ROOT, rel)
    if not os.path.exists(m):
        pytest.skip(f"{label} manifest absent")
    assert not _verify(m, allow), label


def test_A_faithful_a11_unchanged():
    pre = os.path.join(CTL, "final_release_preflight.json")
    if not os.path.exists(pre):
        pytest.skip("preflight absent")
    recorded = json.load(open(pre))["legacy_source_sha256"]
    assert len(recorded) == 10
    for rel, h in recorded.items():
        assert _sha(os.path.join(ROOT, rel)) == h, rel


def test_A_release_never_writes_into_the_legacy_directory():
    for m in FR_MODULES:
        code = _code_only(m)
        assert "shutil.copyfile(src, dst)" in code or "copy_with_provenance" in code
    # the legacy directory holds exactly its original ten files
    assert len(os.listdir(LEGACY)) == 10
    assert not any(n.startswith(("F", "S")) and "_" in n
                   for n in os.listdir(LEGACY))


# ==========================================================  B  matrix ======

@requires_outputs
def test_B_matrix_closure_statuses():
    st = _read(os.path.join(TAB, "analysis_status_summary.tsv"))
    s = dict(zip(st["analysis_id"], st["final_status"]))
    for a in ("A09", "A10", "A11"):
        assert s[a] == "ALREADY_VALIDATED_FORMATTED", a
        assert s[a] != "NEEDS_FORMATTING_ONLY"
    assert s["A19"] == "OPTIONAL_DEFERRED"
    for a in ("A04", "A05", "A06", "A07", "A08", "A12", "A13", "A14",
              "A15", "A16", "A17", "A18"):
        assert s[a] == "ALREADY_VALIDATED", a
    for a in ("A20", "A21", "A22"):
        assert s[a] == "OUT_OF_SCOPE", a


@requires_outputs
def test_B_no_row_regresses_to_needs_computation():
    st = _read(os.path.join(TAB, "analysis_status_summary.tsv"))
    assert "NEEDS_COMPUTATION" not in set(st["final_status"])
    assert set(st["final_status"]) <= {"ALREADY_VALIDATED",
                                       "ALREADY_VALIDATED_FORMATTED",
                                       "OPTIONAL_DEFERRED", "OUT_OF_SCOPE"}


# =============================================  C  formatting non-regression

@pytest.mark.parametrize("row", ["A09", "A10", "A11"])
def test_C_legacy_values_match_the_authoritative_source(row):
    src = os.path.join(LEGACY, fr.LEGACY_TABLES[row])
    rel = os.path.join(FE, "tables", fr.LEGACY_TABLES[row])
    if not os.path.exists(rel):
        pytest.skip("release not generated")
    assert _sha(src) == _sha(rel), "release plotting table must be byte-identical"
    pd.testing.assert_frame_equal(pd.read_csv(src, sep="\t"),
                                  pd.read_csv(rel, sep="\t"), check_exact=True)


@requires_outputs
def test_C_legacy_table_shapes_and_seeds():
    a = fr.legacy_table("A09")
    assert a.shape == (96, 6) and sorted(a["seed"].unique()) == common.SEEDS
    c = fr.legacy_table("A10")
    assert c.shape == (72, 6)          # pooled across seeds by design
    b = fr.legacy_table("A11")
    assert b.shape == (12, 5) and sorted(b["seed"].unique()) == common.SEEDS


def test_C_no_new_estimator_or_filter_in_the_release_code():
    """No estimator is *called*.

    The word "Ridge" legitimately appears as an axis label for a stored A11
    column, so this checks call and import nodes rather than raw text.
    """
    banned = {"Ridge", "permutation_importance", "GridSearchCV", "RidgeCV",
              "train_test_split", "cross_val_score", "ols_slope",
              "hierarchical_bootstrap", "cell_mean_bootstrap", "polyfit",
              "lstsq", "curve_fit"}
    for m in FR_MODULES:
        tree = ast.parse(_code_only(m))
        called, imported = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                called.add(fn.attr if isinstance(fn, ast.Attribute)
                           else getattr(fn, "id", ""))
            elif isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
                imported.update(a.name for a in node.names)
        assert not (banned & called), f"{m} calls {banned & called}"
        assert not (banned & imported), f"{m} imports {banned & imported}"
        assert "fit" not in called, f"{m} calls fit()"


@requires_outputs
def test_C_each_formatted_figure_has_its_plotting_table():
    for row in ("A09", "A10", "A11"):
        stem = fr.LEGACY_STEM[row]
        assert os.path.exists(os.path.join(FE, "figures", f"{stem}.png"))
        assert os.path.exists(os.path.join(FE, "figures", f"{stem}_caption.md"))
        assert os.path.exists(os.path.join(FE, "tables", fr.LEGACY_TABLES[row]))


# ====================================================  D  figure selection ==

@requires_outputs
def test_D_main_figures_exist_in_three_formats_with_captions():
    idx = _read(os.path.join(TAB, "final_figure_index.tsv"))
    main = idx[idx["category"] == "MAIN"]
    assert len(main) == 7
    for _, r in main.iterrows():
        for ext in ("png", "pdf", "svg"):
            p = os.path.join(FIG_M, f"{r['release_stem']}.{ext}")
            assert os.path.exists(p) and os.path.getsize(p) > 0, p
        cap = os.path.join(CAP_M, f"{r['release_stem']}_caption.md")
        assert os.path.exists(cap) and os.path.getsize(cap) > 0


@requires_outputs
def test_D_supplementary_figures_complete():
    idx = _read(os.path.join(TAB, "final_figure_index.tsv"))
    supp = idx[idx["category"] == "SUPPLEMENTARY"]
    assert len(supp) == 12
    for _, r in supp.iterrows():
        for ext in ("png", "pdf", "svg"):
            assert os.path.exists(os.path.join(FIG_S, f"{r['release_stem']}.{ext}"))
        assert os.path.exists(os.path.join(CAP_S,
                                           f"{r['release_stem']}_caption.md"))


@requires_outputs
def test_D_every_release_copy_matches_its_source_hash():
    man = json.load(open(os.path.join(CTL, "final_release_manifest.json")))
    assert man["release_copies"], "copies must be recorded"
    for c in man["release_copies"]:
        assert c["equality"] == "IDENTICAL", c["source_path"]
        assert c["source_sha256"] == c["release_sha256"]
        assert _sha(os.path.join(ROOT, c["release_path"])) == c["release_sha256"]
    assert man["all_copies_identical"] is True
    assert man["sources_moved_or_deleted"] is False


@requires_outputs
def test_D_no_source_figure_overwritten():
    """Release copies live only under final_release/."""
    man = json.load(open(os.path.join(CTL, "final_release_manifest.json")))
    for c in man["release_copies"]:
        assert c["release_path"].startswith(
            "reports/behavioral_wfe_fulllexicon_93a577f/final_release/")
        assert c["source_path"] != c["release_path"]


@requires_outputs
def test_D_index_is_deterministic_and_complete():
    idx = _read(os.path.join(TAB, "final_figure_index.tsv"))
    sel = idx[idx["category"].isin(["MAIN", "SUPPLEMENTARY"])]
    order = sorted(sel.index.tolist())
    assert order == sel.index.tolist(), "index must already be sorted"
    for col in ("purpose", "main_finding", "primary_limitation",
                "reason_for_inclusion"):
        assert sel[col].astype(str).str.len().gt(0).all(), col
    assert set(idx["category"]) <= {"MAIN", "SUPPLEMENTARY", "VALIDATION_ONLY",
                                    "MECHANISM_HANDOFF_ONLY",
                                    "NOT_SELECTED_REDUNDANT"}


@requires_outputs
def test_D_no_composite_figure_created():
    man = json.load(open(os.path.join(CTL, "final_release_manifest.json")))
    assert man["composite_figure_created"] is False


# =========================================  E  faithful / adapted integrity ==

@requires_outputs
def test_E_faithful_and_adapted_never_pooled():
    txt = _norm(_text(*_all_release_docs())).lower()
    assert "never pooled" in txt
    assert "common quantitative" in txt or "common axis" in txt
    for banned in ("pooled importance", "combined feature importance",
                   "average of a11 and a15", "a15 corrects a11",
                   "correction of the faithful"):
        if banned == "correction of the faithful":
            i = txt.find(banned)
            while i >= 0:
                assert any(n in txt[max(0, i - 90):i]
                           for n in ("not ", "never", "no ")), banned
                i = txt.find(banned, i + 1)
        else:
            assert banned not in txt, banned


@requires_outputs
def test_E_source_lexicality_is_not_called_exposure():
    txt = _norm(_text(*_all_release_docs())).lower()
    assert "source labels are not training exposure" in txt
    assert "122" in txt and "9 source" in txt


@requires_outputs
def test_E_clean_confounding_disclosed():
    txt = _norm(_text(*_all_release_docs())).lower()
    assert "perfectly confounded" in txt
    assert "lexicality/exposure" in txt


@requires_outputs
def test_E_pseudowords_never_receive_zipf():
    txt = _norm(_text(*_all_release_docs())).lower()
    assert ("undefined for pseudowords" in txt
            and "never" in txt and "imputed" in txt)
    idx = _read(os.path.join(TAB, "final_table_index.tsv"))
    assert not idx["source_path"].str.contains("pseudo.*zipf", case=False,
                                               regex=True).any()


# ==========================================  F  final scientific wording ====

NEGATIONS = ("no ", "not ", "never", "cannot", "neither", "nothing",
             "does not", "is not", "are not", "without")


def _asserted(txt, phrase, window=170):
    i = txt.find(phrase)
    while i >= 0:
        if not any(n in txt[max(0, i - window):i] for n in NEGATIONS):
            return True
        i = txt.find(phrase, i + len(phrase))
    return False


@requires_outputs
@pytest.mark.parametrize("phrase", [
    "premature eos explains the length effect",
    "premature eos fully explains",
    "morphology is absent from",
    "proves that morphology",
    "zero slope proves",
    "caused by the ltm encoder",
    "caused by the decoder",
    "localized to the gate",
    "we recommend changing",
    "the architecture should be changed",
])
def test_F_forbidden_claims_are_never_asserted(phrase):
    txt = _norm(_text(*_all_release_docs())).lower()
    assert not _asserted(txt, phrase), phrase


@requires_outputs
def test_F_required_disclaimers_present():
    txt = _norm(_text(*_all_release_docs())).lower()
    assert "not a complete explanation" in txt
    assert "structurally unobservable" in txt
    assert "no absence of length information may be inferred" in txt
    assert "do not localize" in txt or "does not localize" in txt
    assert "not a claim that morphology is absent from" in txt or \
           "not a claim that morphology is absent" in txt


@requires_outputs
def test_F_central_result_is_not_oversimplified():
    txt = _norm(_text(*_all_release_docs())).lower()
    assert "not \"ltm always has a length effect\"" in txt or \
           "not “ltm always has a length effect”" in txt or \
           "ltm always has a length effect" in txt
    # and where that phrase appears it must be negated
    assert not _asserted(txt, "ltm always has a length effect")


def test_F_negation_detector_self_check():
    assert _asserted("morphology is absent from the model.", "is absent from")
    assert not _asserted(
        "this is not a claim that morphology is absent from the model.",
        "is absent from")


# ==============================================================  G  SSP ======

@requires_outputs
def test_G_ssp_remains_optional_and_deferred():
    st = _read(os.path.join(TAB, "analysis_status_summary.tsv"))
    assert st[st["analysis_id"] == "A19"]["final_status"].iloc[0] == \
        "OPTIONAL_DEFERRED"
    assert fr.SSP_STATUS == "OPTIONAL_DEFERRED"


@requires_outputs
def test_G_no_ssp_artefact_generated():
    for base, dirs, names in os.walk(Q):
        for n in names + dirs:
            assert "ssp" not in n.lower(), os.path.join(base, n)
            assert "sonority" not in n.lower(), os.path.join(base, n)
    idx = _read(os.path.join(TAB, "final_figure_index.tsv"))
    assert not idx["analysis"].astype(str).str.contains("A19").any()


# ====================================================  H  no inference ======

@pytest.mark.parametrize("module", FR_MODULES)
def test_H_no_torch_or_eval_imports(module):
    tree = ast.parse(open(os.path.join(PKG_DIR, module)).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for banned in ("torch", "models", "evaluate", "train", "sklearn"):
        assert banned not in imported, banned


@pytest.mark.parametrize("module", FR_MODULES)
def test_H_no_checkpoint_load_or_absolute_path(module):
    code = _code_only(module)
    # "checkpoint_path" is a legitimate COLUMN NAME in the checkpoint summary,
    # which reports hashes read from provenance; the ban targets actual loading.
    for banned in ("import torch", "external_eval", "load_model_and_vocab",
                   "torch.load", "state_dict", "load_checkpoint",
                   "/Users/", "/home/", ".pt\"", ".pt'"):
        assert banned not in code, banned


@requires_outputs
def test_H_manifest_declares_no_inference_and_no_new_value():
    man = json.load(open(os.path.join(CTL, "final_release_manifest.json")))
    assert man["model_inference_performed"] is False
    assert man["new_scientific_value_added"] is False
    assert man["ssp_status"] == "OPTIONAL_DEFERRED"


@requires_outputs
def test_H_checkpoint_summary_records_no_load():
    cs = _read(os.path.join(TAB, "checkpoint_summary.tsv"))
    assert len(cs) == 4 and sorted(cs["seed"]) == common.SEEDS
    assert not cs["loaded_in_final_release"].any()
    assert cs["checkpoint_sha256"].str.len().eq(64).all()


# ==============================================================  I  determinism

@requires_outputs
def test_I_indexes_are_deterministic():
    for name in ("final_figure_index.tsv", "final_table_index.tsv",
                 "analysis_status_summary.tsv", "dataset_regime_summary.tsv",
                 "checkpoint_summary.tsv", "key_results_summary.tsv"):
        a = _read(os.path.join(TAB, name))
        b = _read(os.path.join(TAB, name))
        pd.testing.assert_frame_equal(a, b)


@requires_outputs
def test_I_figure_selection_matches_the_frozen_spec():
    spec = fr.load_spec()
    idx = _read(os.path.join(TAB, "final_figure_index.tsv"))
    main = idx[idx["category"] == "MAIN"]["figure_number"].tolist()
    supp = idx[idx["category"] == "SUPPLEMENTARY"]["figure_number"].tolist()
    assert main == [f["number"] for f in spec["main_figures"]]
    assert supp == [f["number"] for f in spec["supplementary_figures"]]
    assert 5 <= len(main) <= 7, "frozen target is 5-7 main figures"


@requires_outputs
def test_I_release_readme_paths_resolve():
    txt = open(os.path.join(Q, "README.md")).read()
    for rel in re.findall(r"`(figures/[^`]+|tables/[^`]+|captions/[^`]+)`", txt):
        p = os.path.join(Q, rel.rstrip("/"))
        assert os.path.exists(p) or os.path.isdir(os.path.dirname(p)), rel


@requires_outputs
def test_I_yair_brief_paths_resolve():
    txt = open(os.path.join(Q, "yair_brief.md")).read()
    rels = re.findall(r"reports/\.\.\./(\S+\.(?:png|md))", txt)
    assert rels, "the brief must cite concrete files"
    for rel in rels:
        p = os.path.join(REPORT, rel)
        assert os.path.exists(p), rel
