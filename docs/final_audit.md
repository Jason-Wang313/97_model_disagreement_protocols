# Final Audit

1. Paper: 97 `model_disagreement_protocols`.
2. Submission-hardening version: v4.
3. Last audit timestamp: 2026-06-14 21:24:22 +01:00.
4. Thesis tested: classify model disagreement and select a robot action protocol.
5. Evidence produced: 5 tasks x 7 disagreement families x 5 splits x 9 methods, seven seeds, ablations, stress sweep, pairwise tests, and failure cases.
6. Terminal decision: KILL_ARCHIVE.
7. Main empirical reason: `failure_aware_rl_recovery` beats the proposed protocol on combined-stress task success, 0.701 vs 0.637, and has lower regret, 0.051 vs 0.111.
8. Ablation reason: `minus_probe_action`, `minus_protocol_cost_model`, and `recovery_only_protocol` match or beat full on success/regret.
9. Hostile prior-work pressure: uncertainty, robust control, diagnostic probing, embodied world models, and failure-aware recovery already occupy the obvious novelty space.
10. Reproducibility: `python src/run_experiment.py` regenerates all v4 CSVs, LaTeX tables, figures, and summary text.
11. Exact Downloads PDF path: `C:/Users/wangz/Downloads/97.pdf`.
12. GitHub URL: https://github.com/Jason-Wang313/97_model_disagreement_protocols
13. Desktop policy: no visible Desktop PDF copy should be made.
