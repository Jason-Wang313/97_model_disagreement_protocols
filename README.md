# 97 Model Disagreement Protocols

Submission-hardening version: v5 expanded audit

Terminal decision: KILL_ARCHIVE for ICLR main.

This repository contains a frozen, CPU-only evidence audit for the claim that robot model disagreement becomes useful when the robot classifies the disagreement type and chooses an action protocol: commit, probe, switch controller, abstain, or recover. The v5 rebuild is substantially stronger than the previous v4.1 archive, but it still does not support a submission-ready positive paper.

## Evidence Summary

The v5 runner builds a deterministic disagreement-to-action benchmark with:

- 6 robotics task families.
- 8 disagreement/failure families.
- 8 stress and shift splits.
- 14 methods, including robust fallback, conformal filtering, failure-aware recovery, recovery-first control, diagnostic probing, active model selection, information-gain probing, v4, v5, and an oracle.
- 10 seeds and 6 episodes per seed/task/family/split/method cell.
- 322,560 main rollout rows.
- 115,200 ablation rollout rows.
- 259,200 stress-sweep rows.
- 138,240 fixed-risk deployment rows.
- 24 negative cases.

The terminal result is negative. On the hard aggregate, `risk_bounded_typed_disagreement_protocol_v5` reaches success 0.59631 and utility 0.20203, while `recovery_first_policy` reaches success 0.68866 and utility 0.37119. V5 also trails `failure_aware_rl_recovery` on regret, trails the safest abstention baseline on unsafe commitment, fails diagnosis and ablation gates, fails maximum-stress and fixed-risk gates, and has no real robot or accepted high-fidelity benchmark evidence.

## Reproduce

```powershell
python src\run_experiment.py
```

Primary evidence files:

- `results/summary.txt`
- `results/rollouts.csv`
- `results/hard_aggregate_metrics.csv`
- `results/pairwise_stats.csv`
- `results/ablation_metrics.csv`
- `results/stress_sweep.csv`
- `results/fixed_risk_metrics.csv`
- `results/failure_cases.csv`

## Build Archive PDF

```powershell
python scripts\generate_manuscript.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
copy main.pdf C:\Users\wangz\Downloads\97.pdf
cd ..
python scripts\validate_submission_artifacts.py
```

Canonical local PDF: `C:/Users/wangz/Downloads/97.pdf`

Validated PDF: 31 pages, SHA256 `CC500D7CCC351CBAB82FC07B729A98CD13A7201487281C316711072E71320B39`.

No PDF should be copied to the visible Desktop.
