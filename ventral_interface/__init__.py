"""VENTRAL SEMANTIC INTERFACE — frozen factorization diagnostic (S0-S3).

Diagnosis only.  A thin wrapper around frozen historical model / evaluator code:

* model reconstruction  -> `scripts.gating_diagnostics.run_gate_route_audit.build_state`
* C retrieval           -> `train_tasks.evaluate_comprehension_subset`,
                           `frozen_probe.encode_all` / `comprehension_metrics`
* Naming                -> `train_tasks.evaluate_naming`
* AR repetition decode  -> `gating_diagnostics.gate_probe.ar_decode_free` /
                           `ar_decode_forced_length`, route "ltm" only

The ONLY experimental variable is the 300-d vector entering the ventral decoder.  It
is supplied by one forward hook on `model.ltm.to_semantic` (see `injection.py`), so
every condition traverses the identical encoder call, `sem_to_h0`, decoder,
`dec_to_premotor`, `motor` and greedy loop.

Contract: paper_programme/ventral_semantic_interface/VENTRAL_INTERFACE_EXPERIMENT_CONTRACT.md
"""

CONDITIONS = ("S0", "S1", "S2", "S3")
CONVENTIONS = ("freear", "canonical")          # primary, secondary
PRIMARY_CONVENTION = "freear"
ROUTE = "ltm"                                    # the only route ever decoded
