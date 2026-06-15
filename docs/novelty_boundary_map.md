# Novelty Boundary Map

## Crowded Territory

- Generic ensemble disagreement.
- Scalar uncertainty thresholds.
- Conformal risk filtering.
- Robust MPC fallback.
- Failure-aware recovery.
- Diagnostic probing for world models.
- Embodied foundation model robustness.

## Boundary Tested

The only potentially novel boundary was a type-to-protocol mechanism: classify the disagreement family and choose commit, probe, switch controller, abstain, or recover.

## What The v4/v4.1 Audit Found

The boundary is not strong enough. The proposed protocol improves recall but does not beat simpler recovery/fallback baselines on the closed-loop metrics that matter for a robot. The failure cases indicate that the label taxonomy adds intervention cost, especially for epistemic and sensor-corruption cases, without enough success gain.

## Boundary Decision

Novelty remains an idea seed, not a submission-ready contribution.
