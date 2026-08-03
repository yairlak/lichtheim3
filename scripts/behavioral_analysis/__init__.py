"""Behavioral WFE analysis package (Lichtheim3, cohort 93a577f).

Inference-free: this package never loads a checkpoint and never calls the
evaluation model.  It consumes the validated production outputs and produces
the publication figures, their exact plotting tables and captions.

Frozen choices live in `common.py` and must not be edited to change a result:
analysis sets, seed policy (all four seeds; 21 never excluded), route
definitions, metric definitions, the hierarchical bootstrap (B = 10,000,
random seed 20260730, 95 % percentile), and the faithful zip-mismatch
serial-position method.

Entry points:
    python -m scripts.behavioral_analysis.build_canonical_table
    python -m scripts.behavioral_analysis.make_figures --out_dir DIR
    python -m scripts.behavioral_analysis.validate_outputs --figures DIR
"""
from __future__ import annotations

__all__ = ["common", "io", "bootstrap", "compute", "plotting"]
