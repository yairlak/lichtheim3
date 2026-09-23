"""READ-ONLY post-execution analysis for Lesioning V2.

Nothing here loads a model, a checkpoint or torch, applies a lesion, or writes
into the scientific result namespace. It reads finalized cell outputs, verifies
their hashes, aggregates the frozen exact-match metric, and writes tables,
figures and reports under an external --out-dir.
"""
