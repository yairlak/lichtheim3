"""Frozen directional-dose mathematics (contract §6, §8, §9).

All interpolation arithmetic is float64; the result is cast EXACTLY ONCE to the dtype of
the native ŝ tensor (float32).  alpha = 0 returns the input tensor object itself.
"""
from __future__ import annotations

import math
from typing import Dict

import torch

from ventral_directional_dose import ALPHAS, EPS_NORM, EPS_ORTHO

ORDINARY = "ORDINARY"
NEAR_COLLINEAR = "NEAR_COLLINEAR_NLERP"


class DoseHardStop(RuntimeError):
    """ZERO_SHAT / ZERO_PROTOTYPE / NEAR_ANTIPODAL — never decoded, never excluded."""

    def __init__(self, kind: str, count: int, detail: str = ""):
        super().__init__(f"HARD STOP {kind}: {count} item(s). {detail}")
        self.kind, self.count = kind, int(count)


def base_geometry(s_hat: torch.Tensor, v: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per-row float64 geometry of native ŝ vs prototype v (both (N, D))."""
    s64 = s_hat.detach().to(torch.float64)
    v64 = v.detach().to(torch.float64)
    ns = torch.linalg.vector_norm(s64, dim=-1)
    nv = torch.linalg.vector_norm(v64, dim=-1)
    zero_shat = ns <= EPS_NORM
    zero_proto = nv <= EPS_NORM
    safe_ns = torch.where(zero_shat, torch.ones_like(ns), ns)
    safe_nv = torch.where(zero_proto, torch.ones_like(nv), nv)
    us = s64 / safe_ns.unsqueeze(-1)
    up = v64 / safe_nv.unsqueeze(-1)
    d = torch.clamp((us * up).sum(-1), -1.0, 1.0)
    r = up - d.unsqueeze(-1) * us
    norm_r = torch.linalg.vector_norm(r, dim=-1)
    theta = torch.arccos(d)
    small = norm_r <= EPS_ORTHO
    return {
        "ns": ns, "nv": nv, "us": us, "up": up, "d": d, "theta": theta, "r": r,
        "norm_r": norm_r,
        "zero_shat": zero_shat, "zero_proto": zero_proto,
        "near_antipodal": small & (d <= 0) & ~zero_shat & ~zero_proto,
        "near_collinear": small & (d > 0) & ~zero_shat & ~zero_proto,
        "ordinary": ~small & ~zero_shat & ~zero_proto,
    }


def assert_no_hard_stops(geom: Dict[str, torch.Tensor]) -> None:
    for key, kind in (("zero_shat", "ZERO_SHAT"), ("zero_proto", "ZERO_PROTOTYPE"),
                      ("near_antipodal", "NEAR_ANTIPODAL")):
        n = int(geom[key].sum())
        if n:
            raise DoseHardStop(kind, n)


def case_labels(geom: Dict[str, torch.Tensor]):
    return [NEAR_COLLINEAR if bool(c) else ORDINARY for c in geom["near_collinear"]]


def dose_vectors(s_hat: torch.Tensor, v: torch.Tensor, alpha: float) -> torch.Tensor:
    """s_alpha = ||ŝ|| * u_alpha, one float32 cast.  alpha = 0 → the input object itself."""
    alpha = float(alpha)
    if alpha == 0.0:
        return s_hat
    if alpha not in ALPHAS:
        raise ValueError(f"alpha {alpha} is not a frozen scientific alpha {ALPHAS}")
    geom = base_geometry(s_hat, v)
    assert_no_hard_stops(geom)
    us, up, theta = geom["us"], geom["up"], geom["theta"]
    ordinary = geom["ordinary"].unsqueeze(-1)
    safe_nr = torch.where(geom["norm_r"] > 0, geom["norm_r"], torch.ones_like(geom["norm_r"]))
    q = geom["r"] / safe_nr.unsqueeze(-1)
    u_slerp = torch.cos(alpha * theta).unsqueeze(-1) * us + torch.sin(alpha * theta).unsqueeze(-1) * q
    u_nlerp = (1.0 - alpha) * us + alpha * up
    u = torch.where(ordinary, u_slerp, u_nlerp)
    u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True)
    s_alpha64 = geom["ns"].unsqueeze(-1) * u
    return s_alpha64.to(s_hat.dtype)                    # the ONE cast


def _unit64(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().to(torch.float64)
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)


def angle_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """atan2(||a − (a·b)b||, a·b) on unit float64 vectors (well-conditioned at 0 and π)."""
    ua, ub = _unit64(a), _unit64(b)
    dot = (ua * ub).sum(-1)
    perp = torch.linalg.vector_norm(ua - dot.unsqueeze(-1) * ub, dim=-1)
    return torch.atan2(perp, dot)


def vector_checks(s_hat: torch.Tensor, v: torch.Tensor, s_alpha: torch.Tensor,
                  alpha: float) -> Dict[str, torch.Tensor]:
    """Frozen per-item verification fields for one alpha (float64 on the float32 values)."""
    geom = base_geometry(s_hat, v)
    ns = geom["ns"]
    na = torch.linalg.vector_norm(s_alpha.detach().to(torch.float64), dim=-1)
    us, ua, up = _unit64(s_hat), _unit64(s_alpha), _unit64(v)
    ang = angle_between(s_hat, s_alpha)
    theta = angle_between(s_hat, v)
    frac = torch.where(theta > EPS_ORTHO, ang / torch.where(theta > 0, theta, torch.ones_like(theta)),
                       torch.full_like(theta, float("nan")))
    err = torch.where(geom["ordinary"], (ang - float(alpha) * theta).abs(),
                      torch.full_like(theta, float("nan")))
    return {
        "s_alpha_norm": na,
        "norm_rel_err": (na / ns - 1.0).abs(),
        "cos_us_ualpha": (us * ua).sum(-1),
        "cos_ualpha_up": (ua * up).sum(-1),
        "angle_from_native_rad": ang,
        "angular_fraction": frac,
        "angle_err_rad": err,
        "theta_atan2": theta,
        "ordinary": geom["ordinary"],
        "near_collinear": geom["near_collinear"],
    }


def is_nan(x: float) -> bool:
    return isinstance(x, float) and math.isnan(x)
