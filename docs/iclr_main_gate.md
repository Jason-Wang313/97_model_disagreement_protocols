# ICLR Main Gate

Paper: 97 `model_disagreement_protocols`

Version: v4

Gate verdict: KILL_ARCHIVE

## Required Gate

The proposed disagreement protocol had to beat the strongest non-oracle baseline on combined-stress task success and unsafe-commit reduction, improve useful-disagreement recall or type accuracy without excessive false alarms/cost, and survive ablations.

## Measured Outcome

- Best non-oracle success baseline: `failure_aware_rl_recovery`, 0.701 task success.
- Proposed protocol: 0.637 task success.
- Best/safest baseline unsafe commits: essentially zero, matched by several methods.
- Proposed protocol regret: 0.111.
- Failure-aware recovery regret: 0.051.
- Ablations matching or beating full: `minus_probe_action`, `minus_protocol_cost_model`, `recovery_only_protocol`.

## Verdict

The proposed mechanism is not submission-ready. It detects many useful disagreements, but the type-to-protocol machinery does not translate into better closed-loop performance than simpler recovery/fallback strategies. Archive rather than submit.
