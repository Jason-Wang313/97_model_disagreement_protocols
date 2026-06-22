# Paper 97 Terminal Audit

Date: 2026-06-22

Paper: `97_model_disagreement_protocols`

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO

## Execution

- Created frozen plan-first expanded document: `docs/paper97_expanded_submission_plan_20260622.md`.
- Compiled experiment script with `python -m py_compile src/run_experiment.py`.
- Reran the full v5 benchmark with six tasks, eight disagreement families, eight splits, fourteen methods, ten seeds, ablations, stress sweep, fixed-risk deployment budgets, paired comparisons, and failure cases.
- Generated manuscript and BibTeX with `python scripts/generate_manuscript.py`.
- Built a 31-page PDF with bright boxed clickable citations.
- Copied final artifact only to `C:/Users/wangz/Downloads/97.pdf`.
- Validated artifact with `python scripts/validate_submission_artifacts.py`.
- Rendered representative pages and visually checked layout, table fit, citation boxes, and bibliography links.

## Evidence

- V5 hard success: `0.59631`.
- Best hard success: `0.68866` from `recovery_first_policy`.
- V5 hard unsafe: `0.11739`.
- Safest hard unsafe: `0.05261` from `variance_threshold_abstention`.
- V5 hard regret: `0.28350`.
- Best hard regret: `0.13682` from `failure_aware_rl_recovery`.
- V5 hard utility: `0.20203`.
- Best hard utility: `0.37119` from `recovery_first_policy`.
- V5 false alarm: `0.00807`.
- Ablations that match or beat the full mechanism: `no_fixed_risk_budget`, `no_probe_action`, `no_value_of_information_gate`, `recovery_only_protocol`, `v4_protocol_rules`.

## Artifact Status

- Canonical PDF: `C:/Users/wangz/Downloads/97.pdf`.
- PDF SHA256: `CC500D7CCC351CBAB82FC07B729A98CD13A7201487281C316711072E71320B39`.
- PDF pages: 31.
- Visible Desktop PDF: absent.
- Public GitHub target: `https://github.com/Jason-Wang313/97_model_disagreement_protocols`.

## Terminal Rationale

Typed disagreement remains an interesting diagnostic idea, but Paper 97 is not an honest ICLR-main submission. Direct recovery and failure-aware baselines dominate the deployment objective, the full mechanism is not ablation-necessary, maximum-stress and fixed-risk gates fail, and the evidence is local CPU-only simulation rather than external robotics validation.
