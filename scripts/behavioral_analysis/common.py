"""Frozen constants and repository-relative path resolution.

Every value here is fixed by the frozen protocol
(outputs/behavioral_wfe_fulllexicon_93a577f/_control/behavioral_analysis_design.json)
and must not be changed to make a result look different.  No absolute user
path appears anywhere: the repository root is derived from this file's location
and can always be overridden by an explicit CLI argument.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------- paths
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(PACKAGE_DIR))

EXPERIMENT = "behavioral_wfe_fulllexicon_93a577f"
OUTPUTS_ROOT = os.path.join(REPO_ROOT, "outputs", EXPERIMENT)
PRODUCTION_ROOT = os.path.join(OUTPUTS_ROOT, "full_wfe_evaluation")
ANALYSIS_ROOT = os.path.join(OUTPUTS_ROOT, "behavioral_analysis")
AUDITS_ROOT = os.path.join(OUTPUTS_ROOT, "audits")
REPORT_ROOT = os.path.join(REPO_ROOT, "reports", EXPERIMENT)

CANONICAL_TABLE = os.path.join(ANALYSIS_ROOT, "tables",
                               "canonical_behavioral_item_table.tsv")
ITEM_MANIFEST = os.path.join(AUDITS_ROOT, "wfe_analysis_item_manifest.tsv")

# ------------------------------------------------------- frozen study design
SEEDS = [19, 20, 21, 22]                  # seed 21 is never excluded
CEILING_SEEDS = [19, 20, 22]              # exact-ceiling sensitivity only
SEED_EPOCHS = {19: 155, 20: 130, 21: 145, 22: 140}
ROUTES = ["full", "wm", "ltm"]
ROUTE_LABEL = {"full": "FULL", "wm": "WM", "ltm": "LTM"}

# WFE lengths: 6 is absent by construction of the stimulus set.
LENGTHS = [3, 4, 5, 7, 8, 9]

TRAINING_COMMIT = "93a577fd9822955fa272ee733fa7e2acf81f1333"
EVALUATION_CODE_COMMIT = "e876b755d0475ed11e5fbc0419a0bd8860dfd325"

ANALYSIS_SETS = [
    "FAITHFUL_WFE_ALL",
    "LICHTHEIM_CLEAN",
    "ALL_WITH_EXPOSURE_STRATA",
    "TRAINED_REAL_FREQUENCY_PRIMARY",
    "TRAINED_REAL_FREQUENCY_SENSITIVITY",
]

EXPOSURE_ORDER = [
    "TRAINED_REAL_EXACT",
    "TRAINED_REAL_PRON_VARIANT",
    "UNTRAINED_REAL",
    "NOVEL_PSEUDOWORD",
    "PSEUDO_TRAINING_WORD",
    "PSEUDO_TRAINING_HOMOPHONE",
]
# Strata this small are descriptive only; never the basis of a claim.
SMALL_STRATUM_MAX_N = 7

CLEAN_EXPOSURES = {"real": "TRAINED_REAL_EXACT", "pseudo": "NOVEL_PSEUDOWORD"}
EXPECTED_CLEAN_COUNTS = {"real": 671, "pseudo": 391, "total": 1062}
EXPECTED_CANONICAL_ROWS = 14400          # 4 seeds x 1200 items x 3 routes
EXPECTED_FAITHFUL_ITEMS = 1200

# --------------------------------------------------------- frozen bootstrap
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260730
BOOTSTRAP_CI_LEVEL = 95
BOOTSTRAP_METHOD = (
    "hierarchical: resample seeds with replacement, then items with "
    "replacement within each analysis-set x stratum cell (cell sizes "
    "preserved); statistic recomputed per replicate and averaged over the "
    "resampled seeds; 95% percentile interval"
)

DECODE_CONVENTION = (
    "deterministic autoregressive, forced-length readout (prediction bounded "
    "by gold target length; terminal insertions unobservable), no noise, no "
    "teacher forcing"
)

# ------------------------------------------------------------- presentation
# Red/blue are RESERVED for lexicality and must not encode any other variable.
LEXICALITY_COLOR = {"real": "red", "pseudo": "blue"}
LEXICALITY_LABEL = {"real": "Real words", "pseudo": "Pseudowords"}
SEED_MARKER = {19: "o", 20: "s", 21: "^", 22: "D"}
# Neutral palette for exposure categories (never red/blue).
EXPOSURE_COLOR = "#4d4d4d"
EXPOSURE_ACCENT = "#8c8c8c"

CLEAN_CAPTION = (
    "Real words were restricted to WFE words encountered during training with "
    "the same phonological form. Pseudowords were restricted to WFE "
    "pseudowords whose phonological form was absent from the training lexicon."
)

GATE_CAPTION = (
    "The gate is a word-level scalar computed for the FULL route only; "
    "WM-only and LTM-only never compute it. Lexical confidence is the top-1 "
    "cosine similarity between the encoded form and the semantic bank. The "
    "gate is not a probability that the stimulus is a word, and lexical "
    "confidence is not phonological similarity."
)

SERIAL_POSITION_METHOD = (
    "faithful zip-mismatch positions (Dager Error_Indices), relative position "
    "(index-1)/(length-1), PCHIP interpolation to 100 points, item-count "
    "weighted across lengths; no Levenshtein alignment is used"
)


def repo_relative(path: str) -> str:
    """Path relative to the repository root, for logs and manifests."""
    return os.path.relpath(path, REPO_ROOT)
