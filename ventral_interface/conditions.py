"""The four semantic input vectors and their vector-level validity gates (B, D).

S0  NATIVE       : the encoder-produced s_hat, passed through unchanged.
S1  RAW_RETRIEVED: bank_raw[top1_idx], top1_idx from the frozen historical C rule.
S2  RAW_TRUE     : bank_raw[item_idx], exactly what historical Naming feeds the decoder.
S3  RADIAL       : ||bank_raw[top1_idx]|| * s_hat / ||s_hat||.

No other manipulation exists in this module, by construction.
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F

#: ||s_hat||_2 (float64 of the float32 vector) at or below which S3 is DEGENERATE.
S3_DEGENERATE_NORM = 1e-6
#: Gate D cosine tolerance: cos(S3, s_hat) >= 1 - GATE_D_COS_TOL.
GATE_D_COS_TOL = 1e-6
#: Gate D ranking tolerance: a top-1/top-2 index change is accepted only as a tie,
#: i.e. when the two candidates' cosines differ by at most this amount.
GATE_D_TIE_TOL = 1e-6


def raw_retrieved(bank_raw: torch.Tensor, top1_idx: Sequence[int]) -> torch.Tensor:
    """S1: the RAW / UNNORMALIZED GloVe row of the retrieved lexical identity."""
    return bank_raw[torch.as_tensor(list(top1_idx), dtype=torch.long)]


def raw_true(bank_raw: torch.Tensor, item_idx: Sequence[int]) -> torch.Tensor:
    """S2: the RAW target GloVe row, gathered by bank index exactly like
    `train_tasks.evaluate_naming` (`bank_raw[torch.tensor(idx)]`)."""
    return bank_raw[torch.as_tensor(list(item_idx), dtype=torch.long)]


def radial(s_hat: torch.Tensor, target_norm: torch.Tensor):
    """S3: rescale s_hat to `target_norm` without changing its direction.

    Computed in float64 and cast back to s_hat's dtype.  Degenerate rows
    (||s_hat|| <= S3_DEGENERATE_NORM) are returned UNCHANGED so the decode stays
    defined, and are flagged; the contract excludes them from every S3 denominator.
    Returns (vector, degenerate_mask).
    """
    s64 = s_hat.detach().to(torch.float64)
    n = torch.linalg.vector_norm(s64, dim=-1)
    deg = n <= S3_DEGENERATE_NORM
    scale = torch.where(deg, torch.ones_like(n),
                        target_norm.detach().to(torch.float64) / torch.where(deg, torch.ones_like(n), n))
    out = (s64 * scale.unsqueeze(-1)).to(s_hat.dtype)
    out = torch.where(deg.unsqueeze(-1), s_hat, out)
    return out, deg


def gate_b(bank_raw: torch.Tensor, item_idx: Sequence[int],
           top1_idx: Sequence[int]) -> Dict[str, object]:
    """GATE B — whenever retrieved lexical identity == target lexical identity,
    S1 and S2 must be exactly equal (same raw bank row)."""
    s1 = raw_retrieved(bank_raw, top1_idx)
    s2 = raw_true(bank_raw, item_idx)
    same = torch.as_tensor([int(a) == int(b) for a, b in zip(item_idx, top1_idx)])
    exact = torch.ones(len(item_idx), dtype=torch.bool)
    maxabs = torch.zeros(len(item_idx), dtype=torch.float64)
    if bool(same.any()):
        d = (s1[same].to(torch.float64) - s2[same].to(torch.float64)).abs().amax(dim=-1)
        maxabs[same] = d
        exact[same] = (s1[same] == s2[same]).all(dim=-1)
    return {
        "per_item_equal_row": same.tolist(),
        "per_item_exact": exact.tolist(),
        "per_item_max_abs_diff": maxabs.tolist(),
        "n_equal_row": int(same.sum()),
        "global_max_abs_diff": float(maxabs.max()) if len(item_idx) else 0.0,
        "pass": bool(exact[same].all()) if bool(same.any()) else True,
    }


def cosine_top2(v: torch.Tensor, bank_n: torch.Tensor, chunk: int = 512):
    """Top-1 / top-2 cosine against the row-normalised bank, historical rule
    (normalise query, dot with normalised bank, argmax).  Top-2 is the argmax after
    masking the top-1 index, so the top-1 index is identical to `argmax`."""
    q_all = F.normalize(v, dim=-1)
    i1, c1, i2, c2 = [], [], [], []
    for lo in range(0, q_all.shape[0], chunk):
        sims = q_all[lo:lo + chunk] @ bank_n.t()
        a1 = sims.argmax(dim=1)
        rows = torch.arange(sims.shape[0])
        v1 = sims[rows, a1]
        sims[rows, a1] = -2.0
        a2 = sims.argmax(dim=1)
        v2 = sims[rows, a2]
        i1.append(a1); c1.append(v1); i2.append(a2); c2.append(v2)
    return torch.cat(i1), torch.cat(c1), torch.cat(i2), torch.cat(c2)


def gate_d(s_hat: torch.Tensor, s3: torch.Tensor, degenerate: torch.Tensor,
           bank_n: torch.Tensor) -> Dict[str, object]:
    """GATE D — S3 preserves the direction of s_hat on every nondegenerate item:
    cos(S3, s_hat) ~ 1, and the cosine ranking against the normalised bank
    (top-1 and top-2 identity) is unchanged up to ties within GATE_D_TIE_TOL."""
    nd = ~degenerate
    cos = F.cosine_similarity(s3.to(torch.float64), s_hat.to(torch.float64), dim=-1)
    a1, c1, a2, c2 = cosine_top2(s_hat, bank_n)
    b1, d1, b2, d2 = cosine_top2(s3, bank_n)
    rank_ok = []
    for k in range(s_hat.shape[0]):
        ok1 = bool(a1[k] == b1[k]) or (
            abs(float(c1[k]) - float(d1[k])) <= GATE_D_TIE_TOL
            and abs(float(c1[k]) - float(c2[k])) <= GATE_D_TIE_TOL)
        ok2 = bool(a2[k] == b2[k]) or abs(float(c2[k]) - float(d2[k])) <= GATE_D_TIE_TOL
        rank_ok.append(bool(ok1 and ok2))
    rank_ok_t = torch.as_tensor(rank_ok)
    cos_ok = cos >= 1.0 - GATE_D_COS_TOL
    return {
        "per_item_cos_s3_shat": cos.tolist(),
        "per_item_s3_top1_index": b1.tolist(),
        "per_item_rank_preserved": rank_ok,
        "n_degenerate": int(degenerate.sum()),
        "min_cos_nondegenerate": float(cos[nd].min()) if bool(nd.any()) else None,
        "n_rank_changed_nondegenerate": int((~rank_ok_t & nd).sum()),
        "pass": bool((cos_ok | degenerate).all() and (rank_ok_t | degenerate).all()),
    }
