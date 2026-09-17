"""VENTRAL SEMANTIC DIRECTIONAL DOSE — frozen diagnostic (norm-preserving SLERP of native ŝ
toward its immutable retrieved raw-GloVe prototype direction).

Contract: paper_programme/ventral_semantic_directional_dose/DIRECTIONAL_DOSE_EXPERIMENT_CONTRACT.md

This package does NOT modify the closed `ventral_interface` package: it imports its frozen
injection context, decoder wrapper, EOS/divergence helpers, route guard and retrieval rule.
"""

#: The ONLY scientific alphas (contract §7).  alpha = 0 is not a scientific condition.
ALPHAS = (0.25, 0.50, 0.75, 1.00)
ALPHA_KEYS = {0.25: "a025", 0.50: "a050", 0.75: "a075", 1.00: "a100"}
CONVENTIONS = ("freear", "canonical")
STATES = ("W3_SRC", "W3_REP", "W4_SRC", "W4_REP")

# Frozen tolerances (contract §8)
EPS_NORM = 1e-6
EPS_ORTHO = 1e-6
TOL_NORM_REL = 1e-6
TOL_ENDPOINT_COS = 1e-6
TOL_ANGLE_RAD = 1e-6
TOL_MONOTONE_RAD = 1e-9
TOL_LIVE_SHAT = 1e-5

GEOMETRY_BATCH = 256        # = DECODE_BATCH, so preflight ŝ has the decode batch shapes
DECODE_BATCH = 256
RETRIEVAL_BATCH = 512
