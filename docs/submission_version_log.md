# Submission Version Log

## v1 - Generated Draft

- Original continuation-batch generated paper and toy single-seed experiment.

## v2 - Submission Hardening

- Added hostile reviewer attack log and response docs.
- Replaced the toy experiment with seven-seed metrics, stronger synthetic baselines, ablations, stress tests, and negative cases.
- Narrowed claims to synthetic diagnostic evidence.
- Terminal decision: WORKSHOP_ONLY.

## v3 - ICLR Main Gate Archive

- Applied the stricter ICLR-main-conference standard.
- Determined that missing real-robot/high-fidelity evidence and template-generated experiments were not enough for submission.
- Terminal decision: KILL_ARCHIVE.

## v4 - Paper-Specific Evidence Audit

- Rebuilt the runner as a disagreement-to-action benchmark.
- Added strong baselines: conformal risk filtering, robust MPC fallback, failure-aware recovery, diagnostic probing, ensemble calibration, proposed protocol, and action oracle.
- Added task/family/split structure, calibration, safety, regret, ablations, pairwise comparisons, stress sweep, figures, and LaTeX result tables.
- Evidence outcome: proposed protocol loses to `failure_aware_rl_recovery` on combined-stress success, 0.637 vs 0.701, and ablations match or beat full on success/regret.
- Terminal decision: KILL_ARCHIVE.
