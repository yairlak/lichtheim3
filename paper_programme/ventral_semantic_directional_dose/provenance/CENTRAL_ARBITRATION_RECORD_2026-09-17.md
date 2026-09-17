# CENTRAL_ARBITRATION_RECORD_2026-09-17 — directional-dose design pass

CENTRAL STEERING arbitration of the design + audit package (design commit `0b758696a99577f9ead43b33c295522dc158dc98`), relayed by the user:

```
DIRECTIONAL_DOSE_CONTRACT_STATUS=ACCEPTED_WITH_MODIFICATION
SLERP_STATUS=ACCEPTED_WITH_MODIFICATION
NEAR_ANTIPODAL_POLICY=HARD_STOP
TRAINING_SUPERVISION_AUDIT_STATUS=ACCEPTED_WITH_QUALIFICATION
YAIR_FLAG_PROVENANCE_STATUS=ACCEPTED_WITH_QUALIFICATION
YAIR_FLAG_IDENTITY=SEMANTIC_ATTRACTOR_FLAG_CONFIRMED
GO_FOR_DIRECTIONAL_DOSE_IMPLEMENTATION=YES
GO_FOR_DIRECTIONAL_DOSE_EXECUTION=YES
GO_FOR_TRAINING=NO
GO_FOR_ARCHITECTURE_CHANGE=NO
```

Binding modifications:
* **Near-antipodal real items:** HARD STOP. The arbitrary-basis fallback is removed.
* **ZERO_SHAT:** HARD STOP, replacing exclusion from denominators.
* **Zero prototype:** HARD STOP (unchanged).
* **Near-collinear NLERP and ordinary float64 SLERP:** accepted.
* **α = 0:** exact tensor short-circuit, DOSE-B only.
* **Mandatory reporting:** real-state geometry preflight; NEAR_COLLINEAR_CASES stratum; α = 1 vs S1 vs S3 factorization; the T vs R steering question.
* **Training audit:** accepted with qualification. It establishes routing, not gradient magnitude or causal effect; the binding boundary is recorded in contract §17b.
* **Yair flag:** confirmed as `semantic_attractor = True/False`. Evidence is substantive but from paraphrased notes, and the wiring is unspecified. Neither the flag nor an attractor is to be implemented.

The contract amendment implementing these points: `DIRECTIONAL_DOSE_EXPERIMENT_CONTRACT.md` (`CONTRACT_STATUS=AMENDED_PER_CENTRAL_NOT_YET_FROZEN`).
