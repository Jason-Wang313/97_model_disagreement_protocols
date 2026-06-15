# Submission Readiness Audit v4.1

Date: 2026-06-15

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO

## Rerun Command

```powershell
python -m py_compile src\run_experiment.py
python src\run_experiment.py
```

The continuation rerun completed successfully and wrote `results/summary.txt` with terminal decision `KILL_ARCHIVE`.

## CSV Integrity

- `metrics.csv`: 45 rows.
- `per_task_family_metrics.csv`: 1575 rows.
- `seed_task_family_metrics.csv`: 11025 rows.
- `pairwise_stats.csv`: 35 rows.
- `ablation_metrics.csv`: 7 rows.
- `ablation_seed_metrics.csv`: 1715 rows.
- `stress_sweep.csv`: 42 rows.
- `stress_sweep_seed_metrics.csv`: 294 rows.
- `failure_cases.csv`: 12 rows.

Coverage: seven seeds (`0` through `6`), five robotics task families, seven disagreement/failure families, five stress splits, nine methods, seven ablations, and six stress levels.

## Main Combined-Stress Gate

The proposed protocol improves diagnostic metrics but loses the closed-loop gate.

| Method | Success | Unsafe commit | Type acc. | Useful recall | False alarm | Cost | Regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| failure_aware_rl_recovery | 0.70093 +/- 0.01658 | 0.00005 | 0.22514 | 0.82305 | 0.29968 | 0.19806 | 0.05130 |
| robust_mpc_fallback | 0.64875 +/- 0.01227 | 0.00014 | 0.15872 | 0.75240 | 0.29984 | 0.16432 | 0.08246 |
| proposed_disagreement_protocol | 0.63692 +/- 0.01303 | 0.00009 | 0.49647 | 0.97169 | 0.17029 | 0.21401 | 0.11064 |

Paired decision rows over 245 seed/task/family groups:

- Proposed versus `failure_aware_rl_recovery` task success: `-0.064007 +/- 0.010534`, winner `failure_aware_rl_recovery`.
- Proposed versus `failure_aware_rl_recovery` regret: `+0.059341 +/- 0.003653`, winner `failure_aware_rl_recovery`.
- Proposed versus `robust_mpc_fallback` task success: `-0.011827 +/- 0.013637`, statistical tie.
- Proposed versus `robust_mpc_fallback` regret: `+0.028186 +/- 0.007988`, winner `robust_mpc_fallback`.
- Proposed versus `conformal_risk_filter` unsafe commit: `+0.000046 +/- 0.000158`, statistical tie rather than a safety win.

## Ablation Gate

The full method is not the best version of itself:

- `recovery_only_protocol`: success `0.64522 +/- 0.00860`, regret `0.07856`.
- `minus_protocol_cost_model`: success `0.64235 +/- 0.01286`, regret `0.09736`.
- `minus_probe_action`: success `0.63474 +/- 0.01082`, regret `0.10289`.
- Full protocol: success `0.63061 +/- 0.01165`, regret `0.11554`.

This is a fatal mechanism-support failure for the type-to-protocol claim.

## Stress Gate

At maximum stress (`1.0`), failure-aware recovery still wins:

- `failure_aware_rl_recovery`: success `0.68595 +/- 0.00921`, regret `0.05004`.
- `robust_mpc_fallback`: success `0.63187 +/- 0.00778`, regret `0.08969`.
- `proposed_disagreement_protocol`: success `0.59812 +/- 0.01307`, regret `0.12889`.

## Terminal Decision

Paper 97 remains `KILL_ARCHIVE`. The diagnostic/taxonomy signal is real, but it does not translate into superior closed-loop task success, safety, or regret, and the ablations show that simpler recovery/cost/probe variants are stronger than the full mechanism.

