# ICLR Main Gate

Status: FAILED

Terminal decision: KILL_ARCHIVE

ICLR main ready: no

## Frozen Gate Results

- Success gate: failed. V5 success 0.59631 trails `recovery_first_policy` 0.68866.
- Safety gate: failed. V5 unsafe 0.11739 trails `variance_threshold_abstention` 0.05261.
- Regret gate: failed. V5 regret 0.28350 trails `failure_aware_rl_recovery` 0.13682.
- Utility gate: failed. V5 robust utility 0.20203 trails `recovery_first_policy` 0.37119.
- Diagnosis gate: failed. V5 accuracy/recall do not clear the frozen gate.
- False-alarm gate: passed. V5 false alarm is 0.00807.
- Ablation gate: failed. Several simplified mechanisms match or beat the full method.
- Stress gate: failed. Maximum-stress success/utility is dominated by `failure_aware_rl_recovery`.
- Fixed-risk gate: failed. Budget 0.05 is dominated by `recovery_first_policy` or insufficient coverage.
- Scope gate: failed. No hardware, accepted high-fidelity benchmark, or trained checkpoint evidence exists.

## Decision

Do not submit this as a positive ICLR-main paper. Archive it as a strong negative result unless external evidence changes the gate outcomes.
