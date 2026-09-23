"""Lossless recovery of the frozen c_top1 correctness bit. READ-ONLY.

FAILURE_CLASS = EXECUTION_OUTPUT_COMPREHENSION_CORRECT_FIELD_ENCODING

What happened, verified against executable code
-----------------------------------------------
`train_tasks.evaluate_comprehension_subset` returns per-item dicts with keys

    position, bank_index, word, freq_rank, n_phonemes, target_rank,
    target_cos, top1, top5, margin, top1_word

and defines the aggregate as `float(np.mean(m["top1"]))`, so the per-item
`top1` IS the binary strict-top-1 correctness indicator. The dict contains
neither `correct` nor `top1_correct`.

Both Lesioning V2 adapters serialized comprehension rows as

    "prediction": str(r.get("top1", r.get("prediction", "")))
    "correct":    int(r.get("correct", r.get("top1_correct", 0)))

so `prediction` received the true correctness bit and `correct` silently fell
back to 0 on every row.

Consequence
-----------
The binary correctness information is preserved LOSSLESSLY in `prediction`, so
c_top1 is recoverable by reading that field. The PREDICTED IDENTITY is not
recoverable: `top1_word` / `top1_idx` were never serialized.

This module recodes ONLY the frozen c_top1 endpoint, only when the stored
`prediction` is exactly "0" or "1", and fails closed otherwise. The metric,
the endpoint and the results files are all unchanged; nothing is recomputed.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Sequence

from paper_programme.lesioning_v2.post_analysis.io_utils import AnalysisError

#: The one endpoint this repair touches, identified by its full triple.
C_TRIPLE = ("comprehension", "full", "STRICT_TOP1_RETRIEVAL")
C_ENDPOINT = "c_top1"
VALID_PREDICTIONS = ("0", "1")

STORED = "STORED_CORRECT"
RECOVERED = "RECOVERED_FROM_SERIALIZED_TOP1"


def is_comprehension_row(row: Dict) -> bool:
    return (row.get("task"), row.get("route"),
            row.get("decoding_convention")) == C_TRIPLE


def effective_correct(row: Dict, counters: Dict = None) -> tuple:
    """Return (effective_correct, correctness_source).

    Non-comprehension rows are returned untouched. Comprehension rows take the
    binary bit from `prediction`; anything that is not exactly "0"/"1" raises.
    """
    src = int(row["correct"])
    if not is_comprehension_row(row):
        return src, STORED

    if "prediction" not in row:
        raise AnalysisError(
            "comprehension row is missing `prediction`; the c_top1 "
            "correctness bit is unrecoverable and analysis fails closed")
    pred = str(row["prediction"])
    if pred not in VALID_PREDICTIONS:
        raise AnalysisError(
            f"comprehension `prediction` is {pred!r}, not a binary "
            "strict-top-1 indicator; analysis fails closed rather than "
            "guessing a correctness value")
    eff = int(pred)
    if counters is not None:
        counters["n_comprehension_rows"] += 1
        counters[f"n_prediction_{eff}"] += 1
        counters[f"n_source_correct_{src}"] += 1
        if src != eff:
            counters["n_source_differs_from_effective"] += 1
    return eff, RECOVERED


def new_counters() -> Dict[str, int]:
    return {"n_comprehension_rows": 0, "n_prediction_0": 0, "n_prediction_1": 0,
            "n_source_correct_0": 0, "n_source_correct_1": 0,
            "n_source_differs_from_effective": 0}


def repair_record(counters: Dict[str, int], recovered_cells: int) -> Dict:
    """The provenance artifact. Descriptive; no acceptance threshold."""
    ok = counters["n_comprehension_rows"] > 0 and (
        counters["n_prediction_0"] + counters["n_prediction_1"]
        == counters["n_comprehension_rows"])
    return {
        "failure_class":
            "EXECUTION_OUTPUT_COMPREHENSION_CORRECT_FIELD_ENCODING",
        "affected_endpoint": C_ENDPOINT,
        "affected_task": "comprehension",
        "affected_triple": list(C_TRIPLE),
        "source_evaluator":
            "scripts.naming_comprehension/train_tasks.py::"
            "evaluate_comprehension_subset",
        "source_semantics":
            "per-item top1 is the binary strict-top1 correctness value; the "
            "aggregate is float(np.mean(m['top1']))",
        "source_per_item_keys": [
            "position", "bank_index", "word", "freq_rank", "n_phonemes",
            "target_rank", "target_cos", "top1", "top5", "margin", "top1_word"],
        "source_lacks_fields": ["correct", "top1_correct"],
        "faulty_serialization": {
            "prediction": "<- per-item top1 (the true correctness bit)",
            "correct": "<- missing correct/top1_correct, fallback 0",
            "adapters": ["execution/evaluators.py", "execution/lesioned_eval.py"],
            "note": "these files are retained unmodified as the historical "
                    "provenance of the encoding bug",
        },
        "recovery_rule": "effective_correct = int(prediction)",
        "recovery_status": "LOSSLESS_FOR_C_TOP1" if ok else "FAIL",
        "PREDICTED_IDENTITY_RECOVERABLE_FROM_ITEMS_JSONL": "NO",
        "predicted_identity_note":
            "top1_word / top1_idx were never serialized, so which lexical "
            "entry was retrieved cannot be recovered; only the correctness "
            "bit can be",
        "C_TOP1_CORRECTNESS_RECOVERABLE": "YES" if ok else "NO",
        "related_observation":
            "the comprehension `target` column also fell back to the bank "
            "index, since the per-item dict has no `target` key; this is an "
            "identifier, not part of the metric, and is left as stored",
        "counts": dict(counters),
        "n_cells_with_recovered_c_top1": recovered_cells,
        "model_rerun": False,
        "lesion_rerun": False,
        "scientific_rng_used": False,
        "original_results_modified": False,
        "metric_changed": False,
        "endpoint_changed": False,
        "endpoints_recoded": [C_ENDPOINT],
    }


def write(path: str, payload: Dict) -> str:
    from paper_programme.lesioning_v2.post_analysis import io_utils
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    return io_utils.sha256_file(path)
