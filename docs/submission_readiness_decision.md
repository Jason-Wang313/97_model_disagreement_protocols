# Submission Readiness Decision

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO.

## Evidence

The v5 rebuild replaced the short v4.1 archive with a 31-page, generated, Downloads-only manuscript and a much larger frozen evidence package. The audit includes strong non-oracle baselines, ablations, paired seed tests, stress sweeps, fixed-risk deployment budgets, negative cases, bright boxed citation links, and row-count validation.

Hard-aggregate headline:

- `recovery_first_policy`: success 0.68866, utility 0.37119.
- `failure_aware_rl_recovery`: success 0.68425, regret 0.13682.
- `typed_disagreement_protocol_v4`: success 0.60975, utility 0.27847.
- `risk_bounded_typed_disagreement_protocol_v5`: success 0.59631, regret 0.28350, utility 0.20203.

The proposed method has low false alarms, but the closed-loop deployment result is not competitive.

## Terminal Reason

The idea is killed by stronger evidence, not by lack of effort. A paper that argues for richer typed disagreement cannot be ICLR-main-target when direct recovery/fallback policies dominate success, regret, robust utility, stress, fixed-risk deployment, and mechanism ablations.

## Revival Condition

Revival would require external evidence: real robot or accepted high-fidelity simulator validation, implemented learned models/checkpoints, and a result showing that the full typed protocol beats recovery-first and failure-aware baselines under fixed safety budgets.
