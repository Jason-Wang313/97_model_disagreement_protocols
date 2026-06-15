# Paper 97 Terminal Audit

Date: 2026-06-15

Paper: `97_model_disagreement_protocols`

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO

## Execution

- Created plan-first continuation document: `docs/paper97_iclr_submission_execution_plan_20260615.md`.
- Compiled experiment script with `python -m py_compile src/run_experiment.py`.
- Reran the full seven-seed benchmark with five tasks, seven disagreement families, five splits, nine methods, ablations, stress sweep, paired comparisons, and failure cases.
- Rerun log: `C:/Users/wangz/robotics_massive_pool_paper_factory/logs/97_model_disagreement_protocols_continuation_rerun_20260615.log`.

## Evidence

The rerun reproduced the negative decision:

- Proposed combined-stress task success: `0.63692 +/- 0.01303`.
- Failure-aware recovery task success: `0.70093 +/- 0.01658`.
- Paired proposed-minus-failure-aware success: `-0.064007 +/- 0.010534`.
- Proposed planning regret: `0.11064`.
- Failure-aware recovery planning regret: `0.05130`.
- Proposed useful-disagreement recall: `0.97169`, but the diagnostic gain does not produce the best closed-loop outcome.
- `recovery_only_protocol`, `minus_protocol_cost_model`, and `minus_probe_action` match or beat the full method on success/regret.

## Artifact Status

- Canonical PDF: `C:/Users/wangz/Downloads/97.pdf`.
- PDF SHA256: `32444680ACFAC9238FC64B5404C93E777E00650B334CF30EFACED6E5A9A6B0BD`.
- PDF size: `275376` bytes.
- Visible Desktop PDF: absent in verification.
- LaTeX/BibTeX logs: clean except harmless `rerunfilecheck` package metadata line.
- GitHub: `https://github.com/Jason-Wang313/97_model_disagreement_protocols`.

## Terminal Rationale

The type-to-protocol taxonomy remains a plausible diagnostic idea, but Paper 97 is not an honest ICLR-main submission. Stronger, simpler recovery/fallback baselines win the main closed-loop gates, and ablations show the full mechanism is not necessary.
