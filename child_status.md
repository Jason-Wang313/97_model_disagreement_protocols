# Child Status 97

Current stage: expanded v5 submission-hardening terminal
Last update: 2026-06-22 08:17:07 +08:00
PDF: C:/Users/wangz/Downloads/97.pdf
PDF SHA256: CC500D7CCC351CBAB82FC07B729A98CD13A7201487281C316711072E71320B39
PDF pages: 31
GitHub: https://github.com/Jason-Wang313/97_model_disagreement_protocols
Submission-hardening version: v5 expanded audit
Terminal decision: KILL_ARCHIVE
ICLR main ready: no

Evidence basis: frozen CPU-only v5 disagreement-to-action benchmark with 6 tasks, 8 disagreement families, 8 splits, 14 methods, 10 seeds, 322,560 main rollout rows, 115,200 ablation rows, 259,200 stress rows, 138,240 fixed-risk rows, paired seed tests, negative cases, generated manuscript tables/figures, bright boxed clickable citation links, Downloads-only PDF validation, and visual PDF QA.

Terminal reason: the v5 typed-disagreement protocol is dominated by recovery-first and failure-aware baselines on the deployment objectives. It fails success, safety, regret, utility, diagnosis, ablation, maximum-stress, fixed-risk, and scope gates.
