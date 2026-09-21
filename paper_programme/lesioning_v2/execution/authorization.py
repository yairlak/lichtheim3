"""Production authorization artifact: schema and fail-closed validation.

Extends the frozen guard with the two bindings CENTRAL added — the exact
execution commit and the cluster run-matrix hash. Lesion science is untouched.

    {
      "token": "CENTRAL_GO_FOR_SCIENTIFIC_EXECUTION_LESIONING_V2",
      "go_for_scientific_execution": true,
      "execution_commit": "<exact commit>",
      "run_matrix_sha256": "cd48e99c...1732a"
    }
"""
from __future__ import annotations

import json
import os
from typing import Dict

from paper_programme.lesioning_v2.lesion_operator import guard

REQUIRED_TOKEN = guard.REQUIRED_TOKEN
AUTH_ENV = guard.AUTH_ENV

CLUSTER_RUN_MATRIX_SHA256 = \
    "cd48e99cc95fe959315b599d3a1ccbfa4093f0fd5ff3aa7cd3c124a41071732a"
SUPERSEDED_RUN_MATRIX_SHA256 = \
    "b4d3c98816b46f59afb3e32f081fa073dd5a48cd27f2279fdaccb1f47ca6887c"

REQUIRED_FIELDS = ("token", "go_for_scientific_execution",
                   "execution_commit", "run_matrix_sha256")

SCHEMA = {
    "token": f"string, must equal {REQUIRED_TOKEN!r}",
    "go_for_scientific_execution": "boolean, must be exactly true",
    "execution_commit": "string, 40-hex git commit; must equal HEAD at run time",
    "run_matrix_sha256": f"string, must equal {CLUSTER_RUN_MATRIX_SHA256!r}",
    "_note": "additional provenance fields are permitted and are non-operative",
}


class AuthorizationRefused(RuntimeError):
    pass


def load(path: str) -> Dict:
    if not path:
        raise AuthorizationRefused(
            f"no authorization artifact; set {AUTH_ENV}. "
            "GO_FOR_SCIENTIFIC_EXECUTION=NO")
    if not os.path.exists(path):
        raise AuthorizationRefused(f"authorization artifact not found: {path}")
    try:
        return json.load(open(path))
    except Exception as exc:
        raise AuthorizationRefused(f"authorization unreadable: {exc}")


def validate(auth: Dict, head_commit: str, matrix_sha256: str) -> None:
    """Fail closed on every mismatch. No partial acceptance."""
    missing = [f for f in REQUIRED_FIELDS if f not in auth]
    if missing:
        raise AuthorizationRefused(f"authorization missing fields: {missing}")
    if auth["token"] != REQUIRED_TOKEN:
        raise AuthorizationRefused("authorization token does not match")
    if auth["go_for_scientific_execution"] is not True:
        raise AuthorizationRefused(
            "go_for_scientific_execution is not exactly true")
    if auth["run_matrix_sha256"] == SUPERSEDED_RUN_MATRIX_SHA256:
        raise AuthorizationRefused(
            "authorization names the SUPERSEDED pre-cluster run matrix")
    if auth["run_matrix_sha256"] != CLUSTER_RUN_MATRIX_SHA256:
        raise AuthorizationRefused(
            f"authorization run_matrix_sha256 {auth['run_matrix_sha256']} is "
            f"not the authoritative {CLUSTER_RUN_MATRIX_SHA256}")
    if matrix_sha256 != CLUSTER_RUN_MATRIX_SHA256:
        raise AuthorizationRefused(
            f"present run matrix {matrix_sha256} is not the authoritative one")
    if auth["execution_commit"] != head_commit:
        raise AuthorizationRefused(
            "authorization is for a DIFFERENT execution commit\n"
            f"  authorised {auth['execution_commit']}\n"
            f"  HEAD       {head_commit}")


def authorize(head_commit: str, matrix_sha256: str) -> Dict:
    auth = load(os.environ.get(AUTH_ENV))
    validate(auth, head_commit, matrix_sha256)
    return auth
