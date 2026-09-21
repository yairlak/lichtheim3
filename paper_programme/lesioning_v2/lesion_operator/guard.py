"""Hard scientific-execution guard.

GO_FOR_SCIENTIFIC_EXECUTION = NO.

The runner REFUSES every nonzero lesion cell on P1-P4 unless CENTRAL has issued
an explicit frozen authorization artifact. This is a code path, not a comment:
there is no flag a hurried operator can pass to bypass it.

An authorization file must contain the exact token below AND name the run
matrix hash it authorises, so a stale authorization cannot license a changed
matrix.
"""
from __future__ import annotations

import json
import os
from typing import Optional

#: CENTRAL must publish this token in the authorization artifact verbatim.
REQUIRED_TOKEN = "CENTRAL_GO_FOR_SCIENTIFIC_EXECUTION_LESIONING_V2"

#: Compiled-in posture. Changing this constant alone still does not authorise a
#: run: the authorization artifact is checked independently.
GO_FOR_SCIENTIFIC_EXECUTION = False

AUTH_ENV = "L3_LESION_V2_AUTHORIZATION"


class ExecutionRefused(RuntimeError):
    """A nonzero lesion execution was attempted without valid authorization."""


def authorization_path() -> Optional[str]:
    return os.environ.get(AUTH_ENV)


def check_authorized(run_matrix_sha256: str) -> None:
    """Raise unless CENTRAL has authorised EXACTLY this run matrix."""
    path = authorization_path()
    if not path:
        raise ExecutionRefused(
            "GO_FOR_SCIENTIFIC_EXECUTION=NO. No authorization artifact.\n"
            f"CENTRAL must issue one and expose it via {AUTH_ENV}.\n"
            "Dry-run and manifest generation remain available (--dry-run).")
    if not os.path.exists(path):
        raise ExecutionRefused(f"authorization artifact not found: {path}")
    try:
        auth = json.load(open(path))
    except Exception as exc:                            # pragma: no cover
        raise ExecutionRefused(f"authorization artifact unreadable: {exc}")
    if auth.get("token") != REQUIRED_TOKEN:
        raise ExecutionRefused("authorization token does not match")
    if auth.get("run_matrix_sha256") != run_matrix_sha256:
        raise ExecutionRefused(
            "authorization is for a DIFFERENT run matrix\n"
            f"  authorised {auth.get('run_matrix_sha256')}\n"
            f"  present    {run_matrix_sha256}")
    if not auth.get("go_for_scientific_execution") is True:
        raise ExecutionRefused("authorization does not set "
                               "go_for_scientific_execution=true")


def is_intact_cell(severity_k: int) -> bool:
    """k=0 is the intact control cell; it is not a lesion and needs no token."""
    return int(severity_k) == 0
