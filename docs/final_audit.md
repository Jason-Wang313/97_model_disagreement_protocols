# Final Audit

1. Paper: 97 `model_disagreement_protocols`.
2. Submission-hardening version: v5 expanded audit.
3. Last audit timestamp: 2026-06-22 08:17:07 +08:00.
4. Thesis tested: classify model disagreement and select a robot action protocol.
5. Evidence produced: 322,560 main rollout rows, 23,040 dataset rows, 53,760 main group rows, 1,120 seed-metric rows, 112 aggregate metric rows, 140 hard seed rows, 14 hard aggregate rows, 96 hard pairwise rows, 115,200 ablation rows, 259,200 stress rows, 138,240 fixed-risk rows, and 24 negative cases.
6. Terminal decision: KILL_ARCHIVE.
7. Main empirical reason: `recovery_first_policy` beats v5 on hard-aggregate success, 0.68866 vs 0.59631, and robust utility, 0.37119 vs 0.20203.
8. Regret reason: `failure_aware_rl_recovery` beats v5 on regret, 0.13682 vs 0.28350.
9. Safety reason: safest deployable baseline reaches unsafe 0.05261 vs 0.11739 for v5.
10. Ablation reason: `no_fixed_risk_budget`, `no_probe_action`, `no_value_of_information_gate`, `recovery_only_protocol`, and `v4_protocol_rules` match or beat the full mechanism on predefined metrics.
11. Scope reason: no real robot, accepted high-fidelity benchmark, or trained checkpoint evidence exists.
12. Reproducibility: `python src/run_experiment.py` regenerates all v5 CSVs, LaTeX tables, figures, and summary text.
13. Exact Downloads PDF path: `C:/Users/wangz/Downloads/97.pdf`.
14. PDF SHA256: `CC500D7CCC351CBAB82FC07B729A98CD13A7201487281C316711072E71320B39`.
15. GitHub URL: https://github.com/Jason-Wang313/97_model_disagreement_protocols
16. Desktop policy: no visible Desktop PDF copy should be made.
