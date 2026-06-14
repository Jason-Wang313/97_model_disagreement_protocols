# Experiment Rigor Checklist

## v4 Completed

- [x] Paper-specific benchmark rather than the old generic branch scaffold.
- [x] Multiple seeds.
- [x] Strong non-oracle baselines.
- [x] Ablations.
- [x] Stress splits and maximum-stress sweep.
- [x] Uncertainty/calibration metrics.
- [x] Safety and regret metrics.
- [x] Pairwise seed/task/family comparisons.
- [x] Failure cases.
- [x] Generated figures and LaTeX result tables.
- [x] Explicit terminal gate in `results/summary.txt`.

## ICLR Main Bar Not Met

- [ ] Real robot validation.
- [ ] High-fidelity simulator benchmark.
- [ ] Trained learned model checkpoint.
- [ ] External benchmark suite comparison.
- [ ] Evidence that full method beats robust fallback and failure-aware recovery.

Decision: fail ICLR-main empirical-rigor gate; archive.
