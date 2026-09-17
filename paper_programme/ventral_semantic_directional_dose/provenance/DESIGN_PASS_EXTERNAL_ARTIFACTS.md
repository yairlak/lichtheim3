# DESIGN_PASS_EXTERNAL_ARTIFACTS

These are NON-SCIENTIFIC verification artifacts from the directional-dose design and audit pass. They are deliberately stored **outside git**, as read-only files (`chmod 444`), because this pass may commit documentation only.

Directory: `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/archives/ventral_directional_dose_design_20260917/`

| file | SHA256 | what it is |
|---|---|---|
| `gradient_routing_probe_synthetic.py` | `440d6e40b9d42476a11a50312dea26b0a38f374c6cbe810dcb2567e380be1df0` | Randomly initialised DualRouteModel at the V6 architecture (`canonical_config`, wm 128 / ltm 512 / 512, gate α 2.0, threshold 0.7). Synthetic phoneme batches and a synthetic 300-d bank. Each V6 loss term is backpropagated alone from `zero_grad(set_to_none=True)`. There is no checkpoint, no lexicon, no optimizer and no parameter update. Run from worktree `wt-ventral-directional-dose` at `0f25b5b8…`. |
| `gradient_routing_probe_synthetic.out.json` | `343c57e5b186d631efa8629d4f2b12f78af000608b24e3073941b93e9f78b6f7` | Its output: per-term YES / NO(None) routing, dL/dc_LTM and dL/dŝ through the gate, gate parameter list (empty), and bank `requires_grad` (False). |
| `slerp_spec_verification_synthetic.py` | `9350d61a06a2ba823aa6272e1def4c65dfa4cb10ed1522c23094a30f890b6985` | Reference evaluation of the preregistered SLERP and fallback formulas on synthetic numpy vectors: 20,000 ordinary 300-d pairs × α ∈ {0.25, 0.5, 0.75, 1}, plus zero-norm, exact/near same-direction and exact/near antipodal cases. There is no model, no checkpoint and no real item. |
| `slerp_spec_verification_synthetic.out.json` | `0d4c0c5139526d65bebf61a1bdbecea9862453d26de1e34097c9746981c36b74` | Its output. Worst ordinary-case errors: norm 4.4e-16 in float64 and 1.2e-8 after the float32 cast; endpoint cosine deficit ≤ 8.9e-16; angle error 3.0e-15 in float64 and 6.4e-9 rad after the cast; 0 monotonicity violations. All fallback checks pass. |

Environment: python 3.11.15, torch 2.12.1, numpy (lichtheim3 conda env), macOS arm64, CPU.

These scripts are **not** the future directional-dose implementation. Neither touches any real scientific output.
