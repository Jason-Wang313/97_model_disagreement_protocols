# Submission Readiness Decision

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO.

## Evidence

The v4/v4.1 rebuild replaced the generic scaffold with an executable disagreement-to-action audit. The audit includes strong baselines, ablations, stress tests, uncertainty/calibration metrics, pairwise seed/task/family comparisons, and failure cases. The 2026-06-15 continuation rerun reproduced the same decision.

Combined-stress headline:

- `failure_aware_rl_recovery`: task success 0.701, regret 0.051.
- `robust_mpc_fallback`: task success 0.649, regret 0.082.
- `proposed_disagreement_protocol`: task success 0.637, regret 0.111.
- `oracle_disagreement_type`: task success 0.718, regret 0.000.

The proposed method has good recall, but it loses the submission-critical closed-loop objective to failure-aware recovery and is not cleanly supported by its ablations.

## Terminal Reason

The idea is not killed because there is no evidence. It is killed because the new evidence is negative for the main claim. A paper that argues for a richer disagreement taxonomy cannot be ICLR-main-target if a simpler recovery baseline achieves better task success and lower regret under the hardest split.

## Revival Condition

Revival would require a substantially stronger empirical result: real robot or high-fidelity simulator validation, implemented learned models, stronger baselines, and evidence that the full type-to-protocol mechanism beats robust fallback and failure-aware recovery on both success and safety.
