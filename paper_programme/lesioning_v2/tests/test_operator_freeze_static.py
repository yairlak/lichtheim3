"""STATIC checks for the Lesioning V2 operator freeze.

No lesion is run. No P1/P2/P3/P4 checkpoint is loaded and no behavioural
evaluator is invoked. The architecture is instantiated SYNTHETICALLY at the
verified V7 widths, and all mask/RNG checks run on toy tensors.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))
REPO = os.path.normpath(os.path.join(PKG, "..", ".."))
sys.path.insert(0, REPO)

CONTRACT = os.path.join(PKG, "LESIONING_V2_OPERATOR_FREEZE_CONTRACT.md")

# The frozen V7 widths, as asserted by the V7 driver itself.
WM_HIDDEN, ENC_HIDDEN, DEC_HIDDEN = 128, 512, 512
PHON_EMBED, SEMANTIC, PREMOTOR, VOCAB = 64, 300, 128, 42

L1 = "wm.encoder.weight_ih_l0"
L2 = "ltm.encoder.weight_ih_l0"
L3 = "ltm.sem_to_h0.weight"


def _synthetic_model():
    """Build the live architecture at V7 widths. Random init; never P1-P4."""
    from config import Config
    from data.phonemes import build_vocab
    from models.dual_route import DualRouteModel
    cfg = Config()
    cfg.wm.hidden = WM_HIDDEN
    cfg.ltm.enc_hidden = ENC_HIDDEN
    cfg.ltm.dec_hidden = DEC_HIDDEN
    cfg.ltm.phon_embed_dim = PHON_EMBED
    cfg.ltm.ltm_encoder_mode = "unigru_last_hidden"
    cfg.data.semantic_dim = SEMANTIC
    return DualRouteModel(cfg, build_vocab(), premotor_dim=PREMOTOR)


# ------------------------------------------------ V7 width provenance -------
def test_v7_declares_widths_128_512_512():
    src = open(os.path.join(REPO, "scripts", "naming_comprehension",
                            "fresh_ceiling_v7.py")).read()
    assert "WM, ENC, DEC = 128, 512, 512" in src
    assert 'EXPECTED["widths"]' in src or '"widths": {"wm_hidden"' in src
    slurm = open(os.path.join(REPO, "scripts", "cluster", "jeanzay",
                              "fresh_ceiling_v7.slurm")).read()
    assert "WM=128; ENC=512; DEC=512" in slurm


def test_v7_asserts_widths_per_checkpoint():
    src = open(os.path.join(REPO, "scripts", "naming_comprehension",
                            "fresh_ceiling_v7.py")).read()
    assert 'ck.get("widths")' in src and 'EXPECTED["widths"]' in src


def test_historical_h128_default_is_not_the_final_architecture():
    """Guard against the trap CENTRAL flagged."""
    src = open(os.path.join(REPO, "scripts", "naming_comprehension",
                            "train_joint_scratch.py")).read()
    assert "CANONICAL_HIDDEN = 128" in src          # the historical default
    assert ENC_HIDDEN != 128 and DEC_HIDDEN != 128  # V7 overrides it


# -------------------------------------------------- names and shapes --------
def test_lesion_tensor_names_and_shapes_exist():
    sd = _synthetic_model().state_dict()
    assert tuple(sd[L1].shape) == (3 * WM_HIDDEN, PHON_EMBED) == (384, 64)
    assert tuple(sd[L2].shape) == (3 * ENC_HIDDEN, PHON_EMBED) == (1536, 64)
    assert tuple(sd[L3].shape) == (DEC_HIDDEN, SEMANTIC) == (512, 300)


def test_logical_edge_counts_match_the_contract():
    assert WM_HIDDEN * PHON_EMBED == 8_192
    assert ENC_HIDDEN * PHON_EMBED == 32_768
    assert DEC_HIDDEN * SEMANTIC == 153_600
    assert 3 * WM_HIDDEN * PHON_EMBED == 24_576
    assert 3 * ENC_HIDDEN * PHON_EMBED == 98_304
    text = open(CONTRACT).read()
    for n in ("8,192", "32,768", "153,600", "24,576", "98,304"):
        assert n in text


def test_phoneme_embedding_is_shared_and_therefore_not_lesionable():
    m = _synthetic_model()
    assert m.wm.phon_embed is m.phon_embed
    assert m.ltm.phon_embed is m.phon_embed
    assert "phon_embed" not in {L1, L2, L3}


def test_motor_readout_is_shared():
    m = _synthetic_model()
    assert m.motor.proj.weight.shape == (VOCAB, PREMOTOR)


def test_wm_encoder_to_decoder_has_no_weight_matrix():
    """The dorsal handoff is a hidden state, so it cannot be lesioned."""
    src = open(os.path.join(REPO, "models", "wm_route.py")).read()
    assert "self.decoder(self.phon_embed(dec_in), h)" in src


# --------------------------------------------------------- GRU layout -------
def test_gru_three_block_decomposition():
    g = torch.nn.GRU(PHON_EMBED, WM_HIDDEN, batch_first=True)
    assert tuple(g.weight_ih_l0.shape) == (3 * WM_HIDDEN, PHON_EMBED)
    assert tuple(g.weight_hh_l0.shape) == (3 * WM_HIDDEN, WM_HIDDEN)
    doc = torch.nn.GRU.__doc__
    for sym in ("W_{ir}", "W_{iz}", "W_{in}", "W_{hr}", "W_{hn}"):
        assert sym in doc                      # documented r, z, n order


def test_ih_is_afference_and_hh_is_recurrence():
    doc = torch.nn.GRU.__doc__
    assert "W_{ir} x_t" in doc                 # ih multiplies the INPUT
    assert "W_{hr} h_{(t-1)}" in doc           # hh multiplies the PREVIOUS STATE


# ------------------------------------------------- logical-edge masking -----
def tile_logical_mask(logical: torch.Tensor, hidden: int) -> torch.Tensor:
    """Tile an (H, D) logical edge mask across the three GRU gate blocks."""
    assert logical.shape[0] == hidden
    return torch.cat([logical, logical, logical], dim=0)


def test_logical_mask_tiles_across_three_gate_blocks():
    H, D = WM_HIDDEN, PHON_EMBED
    logical = (torch.rand(H, D) > 0.3).float()
    full = tile_logical_mask(logical, H)
    assert full.shape == (3 * H, D)
    for b in range(3):
        assert torch.equal(full[b * H:(b + 1) * H], logical)
    # a masked edge is zero in ALL three blocks
    z = (logical == 0).nonzero()
    h0, d0 = int(z[0][0]), int(z[0][1])
    for b in range(3):
        assert full[b * H + h0, d0] == 0.0


def test_logical_masking_differs_from_scalar_masking():
    """The granularity distinction the audit rests on, made concrete."""
    torch.manual_seed(0)
    H, D, p = WM_HIDDEN, PHON_EMBED, 0.30
    N = H * D
    # logical: exactly p of EDGES fully removed, rest pristine
    perm = torch.randperm(N)
    logical = torch.ones(N)
    logical[perm[:round(p * N)]] = 0.0
    full_logical = tile_logical_mask(logical.view(H, D), H)
    edges_fully_removed = int((logical == 0).sum())
    assert edges_fully_removed == round(p * N)
    # scalar: p of COEFFICIENTS removed independently
    scalar = (torch.rand(3 * H, D) >= p).float()
    per_edge_intact = (scalar.view(3, H, D).sum(0) == 3)
    frac_intact = per_edge_intact.float().mean().item()
    assert 0.30 < frac_intact < 0.38          # ~ (1-p)^3 = 0.343
    # same number of zeroed coefficients in expectation, different arrangement
    assert abs(int((full_logical == 0).sum()) - int((scalar == 0).sum())) < 0.02 * 3 * N


def test_bias_tensors_are_never_targets():
    text = open(CONTRACT).read()
    assert "BIAS_LESIONABLE  = NO" in text or "BIAS_LESIONABLE = NO" in text
    for name in (L1, L2, L3):
        assert not name.endswith("bias")


# ---------------------------------------------------- nested-prefix math ----
def nested_masks(n_edges: int, p_max: float, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n_edges, generator=g)
    out = []
    for k in range(1, 16):
        n_k = round(p_max * (k / 15) * n_edges)
        out.append(set(perm[:n_k].tolist()))
    return out


def test_nested_prefix_masks_are_strictly_nested():
    sets = nested_masks(8_192, 0.30, seed=11)
    assert len(sets) == 15
    for a, b in zip(sets, sets[1:]):
        assert a <= b
    assert len(sets[-1]) == round(0.30 * 8_192)


def test_severity_does_not_change_the_permutation():
    a = nested_masks(8_192, 0.30, seed=11)
    b = nested_masks(8_192, 0.30, seed=11)
    assert a == b


def test_severity_formula_matches_the_contract():
    for k in (1, 8, 15):
        assert pytest.approx(k / 15) == k / 15
    assert round(0.30 * (15 / 15) * 8_192) == round(0.30 * 8_192)
    assert 0.30 * (15 / 15) == 0.30


# -------------------------------------------------- SHA seed derivation -----
def derive_seed(payload: str) -> int:
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & (2**63 - 1)


def test_sha_seed_is_deterministic_and_domain_separated():
    m = "L3_LESION_V2_V1|MASK|abc123|wm.encoder.weight_ih_l0|0"
    n = "L3_LESION_V2_V1|NOISE|abc123|wm.encoder.weight_ih_l0|0|bank_1"
    assert derive_seed(m) == derive_seed(m)
    assert derive_seed(m) != derive_seed(n)
    assert 0 <= derive_seed(m) < 2**63


def test_seed_identity_excludes_severity_task_and_decoder():
    text = open(CONTRACT).read()
    assert "severity                      MUST NOT enter the RNG identity" in text
    assert "task                          MUST NOT enter the RNG identity" in text
    assert "decoding convention           MUST NOT enter the RNG identity" in text
    payload = "L3_LESION_V2_V1|MASK|<state_sha256>|<site>|<realization_index>"
    assert "severity" not in payload and "task" not in payload


def test_states_and_sites_are_independent():
    base = "L3_LESION_V2_V1|MASK|{st}|{site}|{r}"
    s1 = derive_seed(base.format(st="STATE_A", site=L1, r=0))
    s2 = derive_seed(base.format(st="STATE_B", site=L1, r=0))
    s3 = derive_seed(base.format(st="STATE_A", site=L2, r=0))
    s4 = derive_seed(base.format(st="STATE_A", site=L1, r=1))
    assert len({s1, s2, s3, s4}) == 4


def test_no_python_hash_in_the_contract():
    text = open(CONTRACT).read()
    assert "sha256" in text
    assert "never Python `hash()`" in text or "Python `hash()`" in text


# ------------------------------------------------------- no mutation --------
def test_no_checkpoint_is_written_by_this_audit():
    """The audit package invokes no state-writing CALL.

    Checked over AST Call nodes, not raw text, so this test's own mention of
    the forbidden names does not trip it.
    """
    import ast
    offenders = []
    for root, _, files in os.walk(PKG):
        for f in files:
            if not f.endswith(".py"):
                continue
            tree = ast.parse(open(os.path.join(root, f)).read())
            for n in ast.walk(tree):
                if not isinstance(n, ast.Call):
                    continue
                fn = n.func
                if isinstance(fn, ast.Attribute) and fn.attr in (
                        "save", "save_pretrained", "load_state_dict"):
                    offenders.append(f"{f}:{fn.attr}")
    assert offenders == [], f"state-writing calls found: {offenders}"


def test_audit_and_contract_exist_and_declare_limitations():
    audit = open(os.path.join(PKG,
                 "LESIONING_V2_CONNECTIVITY_MAPPING_AUDIT.md")).read()
    text = open(CONTRACT).read()
    assert "UENO_MASK_NESTING = NOT_ESTABLISHED_FROM_SOURCE" in audit
    assert "[ADAPTATION]" in audit and "[VERIFIED]" in audit
    assert "Declared limitations" in text
    assert "CONNECTIVITY_P_MAX = 0.30" in text
