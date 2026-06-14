# Reproducibility Checklist

## Reproduces Locally

- [x] `python src/run_experiment.py`
- [x] `results/summary.txt`
- [x] `results/metrics.csv`
- [x] `results/seed_task_family_metrics.csv`
- [x] `results/per_task_family_metrics.csv`
- [x] `results/pairwise_stats.csv`
- [x] `results/ablation_metrics.csv`
- [x] `results/ablation_seed_metrics.csv`
- [x] `results/stress_sweep.csv`
- [x] `results/stress_sweep_seed_metrics.csv`
- [x] `results/failure_cases.csv`
- [x] `results/combined_stress_table.tex`
- [x] `results/ablation_table.tex`
- [x] `results/pairwise_decision_table.tex`
- [x] Figures in `figures/`
- [x] `paper/main.tex`
- [x] Canonical PDF target: `C:/Users/wangz/Downloads/97.pdf`

## Does Not Reproduce

- [ ] Real robot runs.
- [ ] High-fidelity simulator rollouts.
- [ ] Learned model checkpoints.
- [ ] External robot benchmark comparisons.

This is reproducible as a negative local evidence audit, not as a deployable robotics system paper.
