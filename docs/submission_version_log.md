# Submission Version Log

## v4/v4.1

The v4/v4.1 archive created an executable disagreement-to-action benchmark with five tasks, seven disagreement families, five splits, nine methods, seven seeds, ablations, stress sweeps, paired comparisons, and failure cases. It ended KILL_ARCHIVE because failure-aware recovery beat the proposed protocol on combined-stress success and regret.

## v5 Expanded

Date: 2026-06-22

Changes:

- Expanded to six tasks, eight disagreement families, eight splits, fourteen methods, and ten seeds.
- Added v5 risk-bounded typed-disagreement protocol.
- Added hard-aggregate analysis, paired seed tests, ten ablations, ten-level stress sweep, fixed-risk deployment budgets, and 24 negative cases.
- Added manuscript generator and artifact validator.
- Produced a 31-page PDF with bright boxed clickable citations.

Decision: KILL_ARCHIVE.

Reason: v5 is dominated by recovery-first/failure-aware methods on success, regret, utility, stress, fixed-risk deployment, and ablation gates.
