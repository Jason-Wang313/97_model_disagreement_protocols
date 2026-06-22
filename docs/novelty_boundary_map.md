# Novelty Boundary Map

## Crowded Territory

- Generic ensemble disagreement.
- Scalar uncertainty thresholds.
- Conformal risk filtering.
- Robust MPC fallback.
- Failure-aware recovery.
- Recovery-first policies.
- Diagnostic probing for world models.
- Active model selection and information-gain probing.
- Embodied foundation model robustness.

## Boundary Tested

The only potentially novel boundary was a type-to-protocol mechanism: classify the disagreement family and choose commit, probe, switch controller, abstain, or recover.

## What The Expanded v5 Audit Found

The boundary is not strong enough. The v5 protocol reduces false alarms but does not beat simpler recovery/fallback baselines on the closed-loop metrics that matter for a robot. The failure cases indicate that label taxonomy and probing add cost without enough success, regret, or utility gain.

## Boundary Decision

Novelty remains an idea seed, not a submission-ready contribution.
