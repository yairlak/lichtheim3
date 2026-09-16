"""Frozen output schemas: item_level_factorization.tsv, ar_diagnostic_native_freear.tsv,
summary_metrics.json.  Column order and strata definitions are part of the contract."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from typing import Callable, Dict, List, Sequence

from ventral_interface import CONDITIONS, CONVENTIONS

SCHEMA_VERSION = "ventral-interface-schema-v1"

PROVENANCE_COLUMNS = [
    "witness_id", "state_id", "source_or_repaired", "seed", "source_u",
    "source_checkpoint_sha256", "repaired_head_sha256", "reconstructed_state_identity",
]
IDENTITY_COLUMNS = [
    "item_index", "lexical_identity", "target_phonology", "phoneme_length",
    "lexical_frequency", "lexical_rank",
    "in_C_population", "canonical_C_index", "canonical_C_identity",
    "homophone_group", "homophone_group_size", "homophone_group_members",
]
RETRIEVAL_COLUMNS = [
    "retrieved_index", "retrieved_lexical_identity", "retrieved_phonology",
    "C_contract_correct", "lexical_identity_correct", "phonology_correct",
    "retrieved_is_homophone_not_target",
    "cosine_target_shat", "cosine_top1", "cosine_top2", "top2_index",
    "retrieval_margin", "historical_target_margin",
    "shat_norm", "retrieved_raw_glove_norm", "true_raw_glove_norm",
    "s1_equals_s2_row", "s1_s2_max_abs_diff",
    "s3_degenerate", "s3_cos_to_shat", "s3_top1_index",
    "live_shat_vs_retrieval_shat_max_abs_dev",
]
PER_DECODE_FIELDS = [
    "exact_correct", "predicted_phonology", "pred_length", "eos_emitted",
    "first_eos_step", "eos_before_target_length", "eos_after_target_length",
    "terminated_by_cap", "first_divergence_step", "first_divergence_gold_token",
    "first_divergence_pred_token",
]


def decode_columns() -> List[str]:
    return [f"{c}_{v}_{f}" for c in CONDITIONS for v in CONVENTIONS for f in PER_DECODE_FIELDS]


ITEM_LEVEL_COLUMNS = PROVENANCE_COLUMNS + IDENTITY_COLUMNS + RETRIEVAL_COLUMNS + decode_columns()

AR_DIAGNOSTIC_COLUMNS = [
    "state_id", "item_index", "lexical_identity", "target_phonology", "phoneme_length",
    "S0_freear_predicted_phonology",
    "divergence_step", "divergence_is_eos_position", "gold_token", "chosen_token",
    "gold_rank", "gold_logit", "chosen_logit", "margin_chosen_minus_gold",
    "top5_tokens", "top5_logits", "step_vs_goldprefix_logit_max_abs_dev",
    "margin_numerically_ambiguous",
    "n_generated_after_divergence", "n_positional_mismatches",
    "diag_prefix_correction_label", "diag_prefix_correction_corrected_predicted_phonology",
    "diag_prefix_correction_corrected_exact",
    "diag_prefix_correction_corrected_second_divergence_step",
    "diag_prefix_correction_corrected_terminated_by_cap",
]


def write_tsv(path_or_buf, columns: Sequence[str], rows: Sequence[Dict]) -> None:
    """Deterministic TSV: fixed column order, '\\n' line endings, missing -> error."""
    own = isinstance(path_or_buf, str)
    f = open(path_or_buf, "w", newline="", encoding="utf-8") if own else path_or_buf
    try:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(columns)
        for r in rows:
            missing = [c for c in columns if c not in r]
            extra = [k for k in r if k not in columns]
            if missing or extra:
                raise RuntimeError(f"schema violation: missing={missing[:5]} extra={extra[:5]}")
            w.writerow([_fmt(r[c]) for c in columns])
    finally:
        if own:
            f.close()


def _fmt(v) -> str:
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        return repr(v)
    return str(v)


def tsv_bytes(columns, rows) -> bytes:
    buf = io.StringIO()
    write_tsv(buf, columns, rows)
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------- strata
def _i(v) -> int:
    return int(v)


def _native_wrong(r, conv) -> bool:
    return _i(r[f"S0_{conv}_exact_correct"]) == 0


def _c(r):
    return r["C_contract_correct"]


STRATA: Dict[str, Callable] = {
    # population-level
    "ALL_REPETITION_ITEMS": lambda r, v: True,
    "C_POPULATION": lambda r, v: _i(r["in_C_population"]) == 1,
    "NONCANONICAL_HOMOPHONE_MEMBERS": lambda r, v: _i(r["in_C_population"]) == 0,
    "NATIVE_LTM_WRONG": lambda r, v: _native_wrong(r, v),
    # required C x native strata (C defined on the C population only)
    "C_CORRECT_AND_NATIVE_LTM_WRONG":
        lambda r, v: _i(r["in_C_population"]) == 1 and _c(r) == 1 and _native_wrong(r, v),
    "C_WRONG_AND_NATIVE_LTM_WRONG":
        lambda r, v: _i(r["in_C_population"]) == 1 and _c(r) == 0 and _native_wrong(r, v),
    "C_WRONG_BUT_NATIVE_LTM_PHONOLOGY_CORRECT":
        lambda r, v: _i(r["in_C_population"]) == 1 and _c(r) == 0 and not _native_wrong(r, v),
    # disambiguating supplements (contract §8.3)
    "C_WRONG_BUT_RETRIEVED_PHONOLOGY_CORRECT":
        lambda r, v: _i(r["in_C_population"]) == 1 and _c(r) == 0
        and _i(r["phonology_correct"]) == 1,
    "LEXICAL_IDENTITY_CORRECT_AND_NATIVE_LTM_WRONG":
        lambda r, v: _i(r["lexical_identity_correct"]) == 1 and _native_wrong(r, v),
    "LEXICAL_IDENTITY_WRONG_AND_NATIVE_LTM_WRONG":
        lambda r, v: _i(r["lexical_identity_correct"]) == 0 and _native_wrong(r, v),
}

TRANSITIONS = ("WRONG_TO_CORRECT", "CORRECT_TO_WRONG", "CORRECT_TO_CORRECT", "WRONG_TO_WRONG")


def summarize(rows: Sequence[Dict]) -> Dict[str, object]:
    """states -> convention -> stratum -> condition -> counts.  Transitions are
    relative to S0 under the same state, convention and stratum.  S3 denominators
    exclude S3-degenerate items (for S0/S1/S2 they are included)."""
    out: Dict[str, object] = {}
    for sid in sorted({r["state_id"] for r in rows}):
        srows = [r for r in rows if r["state_id"] == sid]
        sd: Dict[str, object] = {}
        for conv in CONVENTIONS:
            cd: Dict[str, object] = {}
            for sname, pred in STRATA.items():
                members = [r for r in srows if pred(r, conv)]
                kd: Dict[str, object] = {}
                for cond in CONDITIONS:
                    use = [r for r in members
                           if not (cond == "S3" and _i(r["s3_degenerate"]) == 1)]
                    n = len(use)
                    ex = sum(_i(r[f"{cond}_{conv}_exact_correct"]) for r in use)
                    entry = {"denominator": n, "exact_count": ex,
                             "exact_proportion": (ex / n) if n else None}
                    if cond == "S0":
                        entry["transitions_vs_S0"] = None
                    else:
                        t = dict.fromkeys(TRANSITIONS, 0)
                        for r in use:
                            a = _i(r[f"S0_{conv}_exact_correct"])
                            b = _i(r[f"{cond}_{conv}_exact_correct"])
                            key = ("CORRECT" if a else "WRONG") + "_TO_" + ("CORRECT" if b else "WRONG")
                            t[key] += 1
                        entry["transitions_vs_S0"] = t
                    kd[cond] = entry
                cd[sname] = kd
            sd[conv] = cd
        out[sid] = sd
    return out


def schema_descriptor() -> Dict[str, object]:
    d = {
        "schema_version": SCHEMA_VERSION,
        "item_level_factorization.tsv": ITEM_LEVEL_COLUMNS,
        "ar_diagnostic_native_freear.tsv": AR_DIAGNOSTIC_COLUMNS,
        "summary_metrics.json": {
            "top_level_keys": ["schema_version", "contract_sha256", "provenance", "gates",
                               "results", "ar_diagnostic_native_freear"],
            "results": "state_id -> convention(freear|canonical) -> stratum -> condition(S0..S3)"
                       " -> {denominator, exact_count, exact_proportion, transitions_vs_S0}",
            "strata": list(STRATA),
            "transitions": list(TRANSITIONS),
            "ar_diagnostic_native_freear": "state_id -> {n_native_freear_failures, "
                                           "divergence_step_counts, divergence_is_eos_position_count, "
                                           "DIAGNOSTIC_ONLY_PREFIX_CORRECTION: {corrected_exact_count}}",
        },
    }
    d["descriptor_sha256"] = hashlib.sha256(
        json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return d
