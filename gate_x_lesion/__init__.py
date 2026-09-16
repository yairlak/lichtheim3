"""GATE x LESION / RECOVERY — the preregistered lesion x fusion experiment.

Contract: `paper_programme/gate_x_lesion_recovery/GATE_X_LESION_EXPERIMENT_CONTRACT.md`
Frozen conditions: `paper_programme/gate_x_lesion_recovery/gxlr_conditions.frozen.json`

Scope is deliberately narrow (contract §2): activation noise on two encoder states, three
frozen severities, two fusion conditions, two decoding conventions.  No connectivity
damage, no training, no attractor, no recovery.

The fusion and decoding machinery is NOT reimplemented here: `gating_diagnostics` is
imported as-is from the frozen GATING lineage, so FIXED05 keeps its already-validated
algebra and both decoders keep their already-validated conventions.
"""
from gate_x_lesion.noise import (EpsilonCache, FROZEN_LAMBDAS, amplitude,  # noqa: F401
                                 base_epsilon, eta, identity_digest, identity_json)
from gate_x_lesion.targets import (FROZEN_ROUTES, ROUTE_TARGETS,  # noqa: F401
                                   assert_site_compatible, get_module, get_target)
from gate_x_lesion.hooks import (RouteNoiseHook, lesioned_route,  # noqa: F401
                                 state_dict_sha256)
from gate_x_lesion.identity import (reconstructed_state_sha256,  # noqa: F401
                                    verify_manifest_state_identity)

__all__ = [
    "EpsilonCache", "FROZEN_LAMBDAS", "FROZEN_ROUTES", "ROUTE_TARGETS",
    "RouteNoiseHook", "amplitude", "assert_site_compatible", "base_epsilon", "eta",
    "get_module", "get_target", "identity_digest", "identity_json", "lesioned_route",
    "reconstructed_state_sha256", "state_dict_sha256",
    "verify_manifest_state_identity",
]
