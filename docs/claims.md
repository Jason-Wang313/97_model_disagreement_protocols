# Claims

## Claim Under Test

Model disagreement is useful for robotics only when the robot can classify the disagreement type and choose an action protocol: commit, probe, switch controller, abstain, or recover.

## Supported By The v5 Audit

- The repository now contains a frozen CPU-only v5 benchmark with six tasks, eight disagreement families, eight splits, fourteen methods, ten seeds, ablations, stress sweeps, fixed-risk deployment budgets, paired seed comparisons, and failure cases.
- V5 reduces harmless-disagreement false alarms on the hard aggregate: 0.00807.
- The rebuilt manuscript is 31 pages, has bright boxed clickable citations, generated tables/figures, and a validated Downloads-only PDF.
- The negative result is reproducible with `python src/run_experiment.py`.

## Not Supported

- V5 does not beat `recovery_first_policy` on hard-aggregate task success: 0.59631 vs 0.68866.
- V5 does not beat `recovery_first_policy` on robust utility: 0.20203 vs 0.37119.
- V5 does not beat `failure_aware_rl_recovery` on regret: 0.28350 vs 0.13682.
- V5 does not improve over the safest deployable baseline on unsafe commitment: 0.11739 vs 0.05261 for `variance_threshold_abstention`.
- Ablations and simpler mechanisms match or beat the full v5 mechanism.
- Maximum-stress and fixed-risk deployment gates fail.
- No real robot, accepted high-fidelity simulator, or trained-model validation is available.

## Submission Claim

The paper is not ICLR-main ready. The correct terminal label is `KILL_ARCHIVE`.
