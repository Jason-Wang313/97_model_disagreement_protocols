# 97 Model Disagreement Protocols

Submission-hardening version: v4

Terminal decision: KILL_ARCHIVE for ICLR main.

This repository is the v4 evidence audit for the claim that robot model disagreement becomes useful when classified into an action protocol: commit, probe, switch controller, abstain, or recover. The audit is paper-specific and executable, but it does not support a main-track submission.

## Evidence Summary

The runner builds a deterministic disagreement-to-action benchmark with:

- 5 robotics task families.
- 7 hidden disagreement/failure families.
- 5 stress splits.
- 9 methods including conformal filtering, robust MPC fallback, failure-aware recovery, diagnostic probing, ensemble calibration, the proposed protocol, and an action oracle.
- 7 seeds and 88 episodes per seed/task/family/method group.
- Ablations, pairwise tests, failure cases, and stress sweeps.

The proposed protocol improves useful-disagreement recall, but it does not clear the submission gate. On combined stress, failure-aware RL recovery reaches higher task success than the proposed protocol, and stripped variants match or beat the full method on success or regret. The honest terminal state is therefore `KILL_ARCHIVE`.

## Reproduce

```powershell
python src\run_experiment.py
```

Primary evidence files:

- `results/summary.txt`
- `results/metrics.csv`
- `results/pairwise_stats.csv`
- `results/ablation_metrics.csv`
- `results/stress_sweep.csv`
- `results/failure_cases.csv`

## Build Archive PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/97.pdf`

No PDF should be copied to the visible Desktop.
