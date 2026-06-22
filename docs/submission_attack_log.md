# Submission Attack Log

## Attack 1: Stronger Baselines

Added robust MPC fallback, conformal filtering, failure-aware recovery, recovery-first policy, active model selection, information-gain probing, v4, v5, and oracle headroom.

Outcome: KILL_ARCHIVE. Recovery-first and failure-aware methods dominate.

## Attack 2: Harder Shifts

Added model-misspecification, controller-instability, combined-stress, and rare-adversarial hard aggregates.

Outcome: KILL_ARCHIVE. V5 hard success is 0.59631 vs 0.68866 for `recovery_first_policy`.

## Attack 3: Mechanism Ablations

Added ten ablations/removals/simplifications.

Outcome: KILL_ARCHIVE. Several simpler mechanisms match or beat full v5.

## Attack 4: Stress Sweep

Added ten stress levels over the combined split.

Outcome: KILL_ARCHIVE. Maximum stress is dominated by `failure_aware_rl_recovery`.

## Attack 5: Fixed-Risk Deployment

Added unsafe-commit budget sweeps.

Outcome: KILL_ARCHIVE. Budget 0.05 is dominated by `recovery_first_policy` or insufficient coverage.

## Attack 6: Manuscript And Artifact Integrity

Generated a 31-page PDF with bright boxed citations, validated row counts, visual QA, and Downloads-only placement.

Outcome: artifact quality improved, but the scientific terminal decision remains KILL_ARCHIVE.
