"""Frozen REAL-WORD primary battery for Lesioning V2.

Recovered from the accepted design record, not from exploratory work:

    REAL_WORD_PRIMARY_BATTERY          = YES
    PSEUDOWORD_PRIMARY_LESION_BATTERY  = NO
    UENO_ADAPTED_NAMING_PRIMARY        = NO
    HF_LF_PRIMARY_ENDPOINT             = NO
    MANUSCRIPT_CLAIM_TARGET            = CORE_DUAL_ROUTE_LESION_CLAIM

Every evaluator is REUSED from the validated V7 pre-lesion pipeline; none is
reimplemented here.

PRIMARY vs DIAGNOSTIC. The core dual-route lesion claim is about what damaging
each site does to the INTACT SYSTEM's behaviour, so the FULL/native readouts
are primary. The route-isolated WM-only and LTM-only readouts are diagnostic
decompositions that explain a primary effect; they are recorded at full
item-level resolution but do not themselves carry the claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Endpoint:
    key: str
    task: str
    route: str
    decoding_convention: str
    population: str
    numerator: str
    denominator: str
    exactness: str
    undefined_handling: str
    tier: str          # "PRIMARY" | "DIAGNOSTIC"


_EXACT_SEQ = ("exact phoneme-ID sequence equality after trimming at first EOS")
_R_POP = "R = 29,571 real words (full frozen repetition population)"
_C_POP = "C = 27,981 canonical phonological targets, retrieved against the "\
         "full 29,571 bank"

PRIMARY: Tuple[Endpoint, ...] = (
    Endpoint("rep_canonical_full", "repetition", "full",
             "CANONICAL_FORCED_LENGTH_AR", _R_POP,
             "items with exact match", "29571", _EXACT_SEQ,
             "no undefined items; every item is scored", "PRIMARY"),
    Endpoint("rep_freear_full", "repetition", "full",
             "GENUINE_FREE_AR", _R_POP,
             "items with exact match", "29571",
             _EXACT_SEQ + "; non-termination within FREE_AR_MAX_STEPS=12 is an error",
             "no undefined items", "PRIMARY"),
    Endpoint("naming_exact", "naming", "full",
             "SEMANTIC_GREEDY_AR_GLOBAL_CAP", "N = 29,571",
             "items with exact match", "29571", _EXACT_SEQ,
             "no undefined items", "PRIMARY"),
    Endpoint("c_top1", "comprehension", "full",
             "STRICT_TOP1_RETRIEVAL", _C_POP,
             "items whose top-1 retrieval is the canonical target", "27981",
             "strict top-1 identity", "no undefined items", "PRIMARY"),
)

DIAGNOSTIC: Tuple[Endpoint, ...] = (
    Endpoint("rep_canonical_wm", "repetition", "wm",
             "CANONICAL_FORCED_LENGTH_AR", _R_POP, "items with exact match",
             "29571", _EXACT_SEQ, "no undefined items", "DIAGNOSTIC"),
    Endpoint("rep_freear_wm", "repetition", "wm", "GENUINE_FREE_AR", _R_POP,
             "items with exact match", "29571", _EXACT_SEQ,
             "no undefined items", "DIAGNOSTIC"),
    Endpoint("rep_canonical_ltm", "repetition", "ltm",
             "CANONICAL_FORCED_LENGTH_AR", _R_POP, "items with exact match",
             "29571", _EXACT_SEQ, "no undefined items", "DIAGNOSTIC"),
    Endpoint("rep_freear_ltm", "repetition", "ltm", "GENUINE_FREE_AR", _R_POP,
             "items with exact match", "29571", _EXACT_SEQ,
             "no undefined items", "DIAGNOSTIC"),
)

ALL_ENDPOINTS = PRIMARY + DIAGNOSTIC

TASKS = ("repetition", "naming", "comprehension")
DECODING_CONVENTIONS = ("CANONICAL_FORCED_LENGTH_AR", "GENUINE_FREE_AR",
                        "SEMANTIC_GREEDY_AR_GLOBAL_CAP", "STRICT_TOP1_RETRIEVAL")

#: Evaluators reused verbatim from the validated V7 pipeline.
EVALUATOR_BINDING = {
    "CANONICAL_FORCED_LENGTH_AR":
        "scripts.evaluate_train_lexicon_ceiling.evaluate_forms_ar",
    "GENUINE_FREE_AR":
        "train_joint_scratch.JointScratchTrainer.free_ar_repetition semantics "
        "(FREE_AR_MAX_STEPS=12, BOS start, first-EOS trim, target length never "
        "terminates decoding)",
    "SEMANTIC_GREEDY_AR_GLOBAL_CAP":
        "scripts.naming_comprehension.train_tasks.evaluate_naming",
    "STRICT_TOP1_RETRIEVAL":
        "scripts.naming_comprehension.train_tasks.evaluate_comprehension_subset",
}

#: Explicitly NOT part of the V2 primary battery.
EXCLUDED = {
    "pseudoword_lesion_battery": "PSEUDOWORD_PRIMARY_LESION_BATTERY=NO",
    "ueno_adapted_naming": "UENO_ADAPTED_NAMING_PRIMARY=NO",
    "hf_lf_split": "HF_LF_PRIMARY_ENDPOINT=NO",
    "seven_panel_replication": "FULL_SEVEN_PANEL_UENO_REPLICATION=NO",
}
