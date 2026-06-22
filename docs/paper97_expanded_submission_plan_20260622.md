# Paper 97 Expanded Submission-Hardening Plan

Date frozen: 2026-06-22

Paper: `97_model_disagreement_protocols`

Repository: `C:/Users/wangz/robotics_massive_pool_paper_factory/97_model_disagreement_protocols`

Canonical PDF target: `C:/Users/wangz/Downloads/97.pdf`

Desktop policy: do not copy `97.pdf` to the visible Desktop.

## Objective

Rebuild Paper 97 into an honest, submission-grade evidence package for the claim that robot model disagreement becomes useful only when the robot classifies the disagreement type and maps it to an action protocol: commit, probe, switch controller, abstain, or recover.

The rebuilt paper must not optimize for pretty results. It must optimize for a result that survives hostile review: strong baselines, stress tests that expose weaknesses, a frozen protocol, honest reporting of all predefined results, and an explicit terminal decision.

## Current Failure To Attack

The v4.1 archive is a useful executable audit, but it is not submission-ready:

- `failure_aware_rl_recovery` beats the proposed protocol on combined-stress success.
- Robust/failure-aware methods match or beat the proposed protocol on regret and safety.
- `minus_probe_action`, `minus_protocol_cost_model`, and `recovery_only_protocol` undermine mechanism necessity.
- The PDF is only 4 pages and does not contain a full theory, related-work pressure, fixed-risk deployment audit, or 25+ page submission-style manuscript.
- Evidence remains local and CPU-only; no real robot, accepted high-fidelity benchmark, or trained checkpoint is claimed.

## Frozen V5 Experimental Design

### Main Benchmark

- Seeds: 10.
- Tasks: 6 robotics task families.
- Disagreement/failure families: 8.
- Splits: 8, including nominal, single-axis shifts, hard combined stress, and rare/adversarial stress.
- Methods: 14 total, including the v4 proposal, a v5 risk-bounded typed-disagreement protocol, simple uncertainty baselines, conformal filters, robust MPC fallback, failure-aware recovery, diagnostic probing, active model selection, information-gain probing, recovery-first policy, and an oracle.
- Episodes per seed/task/family/split/method cell: 6.
- Expected main rollout rows: 322,560.

### Strong Baselines

The paper may not claim contribution unless the proposed protocol is compared against:

- Mean or variance ensemble policies.
- Variance-threshold abstention.
- Conformal risk filtering.
- Robust MPC fallback.
- Failure-aware recovery.
- World-model diagnostic probing.
- Ensemble calibration.
- Bayesian or active model selection.
- Information-gain probing.
- Recovery-first action policies.
- The previous v4 protocol.
- A non-deployable oracle to expose headroom.

### Required Analyses

- Aggregate metrics by split and method.
- Seed-level metrics for paired tests.
- Hard-aggregate analysis over the hardest splits.
- Pairwise seed tests against the strongest non-oracle baselines.
- Ablation suite with at least 10 removals/simplifications.
- Six-to-ten level stress sweep over the hard combined split.
- Fixed-risk deployment budget sweep.
- Negative-case table with at least 24 representative failures.
- Generated LaTeX tables and figures used directly by the manuscript.

## Frozen Gates

The paper can only improve to `STRONG_REVISE` if the frozen evidence supports all empirical gates:

- Success gate: v5 beats the strongest non-oracle method on hard-aggregate task success with a positive paired lower bound.
- Safety gate: v5 beats or is statistically tied with the safest deployable baseline while also improving success.
- Regret gate: v5 has lower regret than robust fallback and failure-aware recovery, or a statistically tied regret with higher success and utility.
- Diagnosis gate: v5 has top-tier disagreement-type accuracy and useful-disagreement recall.
- False-alarm gate: v5 does not convert harmless disagreement into excessive intervention cost.
- Utility gate: v5 has the best or statistically tied best robust utility under hard stress.
- Ablation gate: the full mechanism beats every component removal on success/regret/utility.
- Stress gate: the advantage does not disappear at maximum stress.
- Fixed-risk gate: at fixed unsafe-commit budgets, v5 has nonzero useful coverage and competitive success.
- Scope gate: ICLR-main readiness remains `no` unless external robot, accepted high-fidelity benchmark, or trained-model evidence exists.

If any empirical gate fails, the terminal decision remains `KILL_ARCHIVE`. If empirical gates pass but scope still lacks external validation, the decision may be `STRONG_REVISE` but ICLR-main readiness remains `no`.

## Theory To Add

The manuscript must add non-filler theory:

- A decision-theoretic decomposition showing when disagreement labels have positive value over direct recovery/fallback.
- A cost-sensitive protocol-selection objective with explicit intervention, probing, unsafe-commit, and regret terms.
- A fixed-risk deployment framing that separates classifier quality from deployable coverage.
- Failure propositions explaining why high type accuracy can still hurt closed-loop utility.
- A reviewer-facing limitation theorem: taxonomy can be dominated by recovery-first control when label value is lower than action cost.

## Manuscript Requirements

- At least 25 pages without padding.
- Bright boxed clickable citation links in the PDF.
- Bibliography sourced from the existing shared literature pool artifacts.
- Tables/figures generated from the frozen CSV outputs, not hand-entered.
- Explicit limitations, negative cases, and terminal decision.
- No claims of hardware, high-fidelity simulation, or trained neural checkpoints unless artifacts exist.

## Artifact Requirements

- Compile the runner: `python -m py_compile src/run_experiment.py`.
- Run the frozen CPU-only experiment: `python src/run_experiment.py`.
- Generate `paper/main.tex` and `paper/references.bib`.
- Build PDF with `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`.
- Copy only the final numbered PDF to `C:/Users/wangz/Downloads/97.pdf`.
- Verify `C:/Users/wangz/Desktop/97.pdf` is absent.
- Validate page count, SHA256, key CSV row counts, and citation-box settings.
- Render representative PDF pages for visual QA.
- Update child docs and root ledgers honestly.
- Commit and push the public GitHub repository.

## Stop Condition Before Paper 98

Do not move to Paper 98 until:

- Paper 97 has a frozen plan document, v5 runner, final docs, 25+ page PDF, and public GitHub push.
- The local child repo is clean and `HEAD` matches `origin/main`.
- `GLOBAL_POOL_STATUS.md`, `BATCH_STATUS.md`, `SUBMISSION_STATUS.md`, `MASTER_REPORT.md`, `MASTER_SUBMISSION_REPORT.md`, and `SUBMISSION_AUDIT_MATRIX.csv` reflect the final Paper 97 decision.
- The Downloads-only PDF path and SHA256 have been verified.
