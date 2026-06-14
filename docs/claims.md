# Claims

## Claim Under Test

Model disagreement is useful for robotics only when the robot can classify the disagreement type and choose an action protocol: commit, probe, switch controller, abstain, or recover.

## Supported By The v4 Audit

- The executable benchmark tests five robot task families, seven hidden disagreement families, five stress splits, nine methods, seven seeds, ablations, stress sweeps, and failure cases.
- The proposed protocol obtains higher useful-disagreement recall than most baselines on combined stress.
- Type-aware intervention reduces the worst unsafe-commit behavior of naive mean and variance policies.

## Not Supported

- The proposed protocol does not beat `failure_aware_rl_recovery` on combined-stress task success: 0.637 vs 0.701.
- The proposed protocol does not improve over the safest baseline on unsafe commits; conformal filtering and several robust methods already reach essentially zero unsafe commits.
- `minus_probe_action`, `minus_protocol_cost_model`, and `recovery_only_protocol` match or beat the full protocol on success or regret.
- No real robot or high-fidelity simulator validation is available.

## Submission Claim

The paper is not ICLR-main ready. The correct terminal label is `KILL_ARCHIVE`.
