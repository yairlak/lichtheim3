"""Frozen Lesioning V2 site definitions.

Each site binds exactly ONE afferent connectivity tensor and ONE activation
target. Recurrent weights, biases and the shared phoneme embedding are never
touched (operator contract section 1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

WM_HIDDEN, ENC_HIDDEN, DEC_HIDDEN = 128, 512, 512
PHON_EMBED, SEMANTIC = 64, 300


@dataclass(frozen=True)
class Site:
    name: str
    connectivity_param: str
    expected_shape: Tuple[int, int]
    kind: str                     # "gru_ih" | "linear"
    hidden: int                   # H for gru_ih; out_features for linear
    input_dim: int
    activation_target: str
    description: str

    @property
    def n_logical_edges(self) -> int:
        return self.hidden * self.input_dim

    @property
    def n_scalar_coefficients(self) -> int:
        return (3 if self.kind == "gru_ih" else 1) * self.n_logical_edges


L1 = Site(
    name="L1",
    connectivity_param="wm.encoder.weight_ih_l0",
    expected_shape=(3 * WM_HIDDEN, PHON_EMBED),
    kind="gru_ih", hidden=WM_HIDDEN, input_dim=PHON_EMBED,
    activation_target="wm.encoder.h_n",
    description="dorsal/WM encoder: embedded phoneme input -> dorsal store",
)
L2 = Site(
    name="L2",
    connectivity_param="ltm.encoder.weight_ih_l0",
    expected_shape=(3 * ENC_HIDDEN, PHON_EMBED),
    kind="gru_ih", hidden=ENC_HIDDEN, input_dim=PHON_EMBED,
    activation_target="ltm.encoder.h_n",
    description="ventral/LTM encoder: phoneme input -> ventral encoding, "
                "perturbed BEFORE to_semantic",
)
L3 = Site(
    name="L3",
    connectivity_param="ltm.sem_to_h0.weight",
    expected_shape=(DEC_HIDDEN, SEMANTIC),
    kind="linear", hidden=DEC_HIDDEN, input_dim=SEMANTIC,
    activation_target="ltm.decoder.h0_post_tanh",
    description="ventral production: semantic pattern -> initial production "
                "state h0 = tanh(sem_to_h0(s_hat))",
)

SITES = {s.name: s for s in (L1, L2, L3)}
SITE_ORDER = ("L1", "L2", "L3")

#: Tensors that must remain bit-identical under every lesion cell.
NEVER_LESIONED = (
    "wm.encoder.weight_hh_l0", "wm.encoder.bias_ih_l0", "wm.encoder.bias_hh_l0",
    "ltm.encoder.weight_hh_l0", "ltm.encoder.bias_ih_l0", "ltm.encoder.bias_hh_l0",
    "ltm.sem_to_h0.bias",
    "ltm.decoder.weight_ih_l0", "ltm.decoder.weight_hh_l0",
    "phon_embed.weight", "wm.phon_embed.weight", "ltm.phon_embed.weight",
    "ltm.to_semantic.0.weight", "ltm.to_semantic.2.weight",
    "motor.proj.weight",
)

#: Required shapes for the first cluster gate (contract section 3).
REQUIRED_SHAPES = {
    "wm.encoder.weight_ih_l0": (3 * WM_HIDDEN, PHON_EMBED),
    "wm.encoder.weight_hh_l0": (3 * WM_HIDDEN, WM_HIDDEN),
    "ltm.encoder.weight_ih_l0": (3 * ENC_HIDDEN, PHON_EMBED),
    "ltm.encoder.weight_hh_l0": (3 * ENC_HIDDEN, ENC_HIDDEN),
    "ltm.sem_to_h0.weight": (DEC_HIDDEN, SEMANTIC),
    "ltm.sem_to_h0.bias": (DEC_HIDDEN,),
    "ltm.decoder.weight_hh_l0": (3 * DEC_HIDDEN, DEC_HIDDEN),
    "phon_embed.weight": (42, PHON_EMBED),
}
