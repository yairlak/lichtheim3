"""RECONSTRUCTED_STATE_SHA256 — the stable pre-lesion identity of a witness.

CLOSURE PASS, O-1.  This module gives an explicit name and a single definition to
the state identity that was previously computed inline in the runner.  **The
derivation is unchanged**: the value is byte-for-byte what the DESIGN freeze pinned
in `checkpoint_manifest.proposed.tsv` and what `IMPLEMENTATION_COMMIT` already used.
Changing the derivation would change every `epsilon`, so it is not changed here.

Why a dedicated identity exists at all.  W3 and W4 are not standalone checkpoint
files (finding F-3).  Each scientific witness is

    BASE TRAINING CHECKPOINT  +  REPAIRED HEAD

reconstructed by `frozen_head_probe._isolated_model(tr, "p_last_hinge", head["state"])`.
There is therefore no single file whose SHA256 identifies the witness, and CENTRAL's
RNG identity slot `checkpoint_sha256` needs a well-defined filler.

Definition
----------

    RECONSTRUCTED_STATE_SHA256
        = sha256( "gxlr-state-v1|" + base_artifact_sha256 + "|" + applied_head_sha256 )

serialised as ASCII, where both operands are the lowercase hex SHA256 of the two
artifact FILES as pinned in the frozen GATING manifest.  For a state with no applied
head the head operand is the empty string.

Properties — each asserted by `tests/test_gate_x_lesion_closure.py::test_O1_*`
-----------------------------------------------------------------------------

1. **It is a pure function of two file digests.**  Nothing else is hashed: not model
   weights, not activations, not `epsilon`, not a lesion tensor, not `lambda`, not the
   fusion condition, not the decoding convention, not the AR prefix, not batch
   position, not process identity.  The function below takes exactly two arguments,
   so no other quantity *can* reach it.

2. **It is computed strictly before any lesion.**  The runner resolves it from the
   manifest immediately after reconstruction and before `measure_intact_sd`, and it is
   hoisted out of the route / lambda / seed loops.

3. **It is invariant** across lesion seeds, severities, routes, items, NATIVE/FIXED05
   and canonical/free-AR: none of those is an input.

4. **It is stable across repeated reconstructions**, because the two artifact files are
   opened read-only and their hashes are re-verified before use.

5. **Nothing downstream can feed back into it.**  `epsilon` is derived FROM this value;
   the dependency is one-way by construction.

Note on earlier wording.  The previous handoff listed O-1 as "composite `state_sha256`
rule — confirm; changes every `epsilon`".  That meant: *if CENTRAL were to choose a
different composition rule, every `epsilon` would change*, i.e. the rule is consequential
and should be confirmed before execution.  It did **not** mean that the identity varies
with `epsilon`.  It cannot; see property 1.
"""
from __future__ import annotations

import hashlib
import re

#: Domain separation tag.  Part of the frozen derivation — changing it changes every
#: epsilon and requires a formal CENTRAL amendment.
STATE_IDENTITY_DOMAIN = "gxlr-state-v1|"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def reconstructed_state_sha256(base_artifact_sha256: str,
                               applied_head_sha256: str = "") -> str:
    """The stable pre-lesion identity of a reconstructed witness.

    Takes exactly two file digests and nothing else.  There is deliberately no
    parameter by which a lesion, a severity, a fusion condition or any runtime state
    could influence the result.
    """
    base = str(base_artifact_sha256).strip().lower()
    head = str(applied_head_sha256 or "").strip().lower()

    if not _HEX64.match(base):
        raise ValueError(
            f"base_artifact_sha256 must be 64 lowercase hex characters, got {base!r}")
    if head and not _HEX64.match(head):
        raise ValueError(
            f"applied_head_sha256 must be 64 lowercase hex characters or empty, "
            f"got {head!r}")

    return hashlib.sha256(
        (STATE_IDENTITY_DOMAIN + base + "|" + head).encode("ascii")).hexdigest()


def verify_manifest_state_identity(row: dict) -> str:
    """Recompute a manifest row's identity and hard-stop if it disagrees.

    The manifest column is authoritative for provenance; this recomputation makes a
    silently edited manifest fail closed rather than quietly re-seed the experiment.
    """
    expected = str(row["state_sha256"]).strip().lower()
    actual = reconstructed_state_sha256(row["base_artifact_sha256"],
                                        row.get("applies_head_sha256", ""))
    if actual != expected:
        raise RuntimeError(
            f"HARD STOP: RECONSTRUCTED_STATE_SHA256 mismatch for "
            f"{row.get('state_id', '?')}: manifest says {expected}, recomputed "
            f"{actual}. The manifest or an artifact hash has been edited; every "
            f"epsilon depends on this value, so the run is refused.")
    return actual
