# Submission Readiness Audit v5

Date: 2026-06-22

Decision: KILL_ARCHIVE

ICLR main ready: no

## Evidence Package

- Main rollout rows: 322,560.
- Dataset rows: 23,040.
- Main group rows: 53,760.
- Main seed metric rows: 1,120.
- Main aggregate rows: 112.
- Hard seed rows: 140.
- Hard aggregate rows: 14.
- Hard pairwise rows: 96.
- Ablation rollout rows: 115,200.
- Stress rollout rows: 259,200.
- Fixed-risk rollout rows: 138,240.
- Negative cases: 24.
- PDF pages: 31.
- PDF SHA256: `CC500D7CCC351CBAB82FC07B729A98CD13A7201487281C316711072E71320B39`.

## Main Finding

The v5 method improves the audit but not the claim. It reaches hard success 0.59631 and utility 0.20203. `recovery_first_policy` reaches hard success 0.68866 and utility 0.37119. `failure_aware_rl_recovery` has lower regret, 0.13682 vs 0.28350.

## Gate Result

Only the false-alarm gate passes. All other empirical gates and the scope gate fail.

## Honest Submission Status

Do not submit as a positive ICLR-main paper. Archive unless external robot or high-fidelity evidence reverses the deployment gates.
