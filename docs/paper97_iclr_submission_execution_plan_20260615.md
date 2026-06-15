# Paper 97 ICLR-Main Continuation Execution Plan

Date: 2026-06-15

Paper: `97_model_disagreement_protocols`

Repository: `C:/Users/wangz/robotics_massive_pool_paper_factory/97_model_disagreement_protocols`

Canonical PDF target: `C:/Users/wangz/Downloads/97.pdf`

Desktop policy: no copy of `97.pdf` may exist on the visible Desktop.

## Objective

Re-audit Paper 97 as an ICLR-main-target candidate from fresh evidence. The paper can move upward only if the rerun proves that classifying robot-model disagreement into action protocols improves closed-loop performance beyond strong recovery, fallback, diagnostic-probing, calibration, and conformal baselines.

## RAM-Light Execution Policy

- Run only this paper's experiment process during the Paper 97 audit.
- Do not reduce seeds, episodes, task families, disagreement families, methods, ablations, or stress levels to save RAM.
- Keep thread fan-out low where possible and stream command output to the root `logs/` directory.

## Evidence To Rerun

1. Compile the experiment script with `python -m py_compile src/run_experiment.py`.
2. Rerun `python src/run_experiment.py` from the child repository.
3. Confirm regenerated outputs:
   - `results/metrics.csv`
   - `results/per_task_family_metrics.csv`
   - `results/seed_task_family_metrics.csv`
   - `results/pairwise_stats.csv`
   - `results/ablation_metrics.csv`
   - `results/ablation_seed_metrics.csv`
   - `results/stress_sweep.csv`
   - `results/stress_sweep_seed_metrics.csv`
   - `results/failure_cases.csv`
   - LaTeX tables and figures consumed by `paper/main.tex`

## Required ICLR-Main Gates

The paper may only become `STRONG_REVISE` if the rerun supports all of the following:

- Proposed disagreement protocol beats the strongest non-oracle combined-stress task-success baseline.
- Proposed protocol beats or decisively improves over the safest non-oracle baseline on unsafe commits, not merely ties at near-zero unsafe rate.
- Proposed protocol has top-tier useful-disagreement recall without excessive noise false alarms.
- Proposed protocol has lower or statistically competitive regret compared with robust fallback and failure-aware recovery.
- Full method beats all ablations on the main success/regret mechanism gate.
- Maximum-stress sweep does not reverse the claimed advantage.
- Manuscript and docs honestly state the local-simulation evidence limit and hostile prior-work boundary.

## Terminal Failure Criteria

Keep or mark `KILL_ARCHIVE` if any of these remain true:

- Failure-aware recovery, robust fallback, or another non-oracle baseline wins success or regret under combined stress.
- Safest baseline ties or beats the proposed method on unsafe commits.
- The proposed method's type/recall gains do not translate into closed-loop gains.
- Ablations such as `minus_probe_action`, `minus_protocol_cost_model`, or `recovery_only_protocol` match or beat the full method.
- Evidence remains local deterministic simulation without real robot or high-fidelity validation.

## Artifact Updates

After rerun and audit:

- Add a continuation audit document with row counts, coverage, decision metrics, paired results, ablations, and terminal decision.
- Update child docs and manuscript only to match what the rerun proves.
- Rebuild `paper/main.pdf` through `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`.
- Copy only the numbered PDF to `C:/Users/wangz/Downloads/97.pdf`.
- Commit and push the child repository to the matching public GitHub repo.
- Update root ledgers: `GLOBAL_POOL_STATUS.md`, `BATCH_STATUS.md`, `SUBMISSION_STATUS.md`, `MASTER_REPORT.md`, `MASTER_SUBMISSION_REPORT.md`.

## Stop Condition Before Moving To Paper 98

Paper 97 is terminal only after all of the following are verified:

- Child repository is clean and local `HEAD` equals `origin/main`.
- GitHub repo is public.
- `C:/Users/wangz/Downloads/97.pdf` exists with recorded SHA256 and size.
- `C:/Users/wangz/Desktop/97.pdf` is absent.
- LaTeX/BibTeX logs have no substantive warnings/errors.
- Root ledgers agree with the child decision, PDF artifact, and GitHub URL.
