# Paper 97 Rebuild Plan: Model Disagreement Protocols

Timestamp: 2026-06-14 20:50:00 +01:00

## Starting Point

Paper 97 is currently a v3 archive. The original research bet is:

> Define action protocols that distinguish useful model disagreement from noise.

The current repo contains a generic synthetic probability scaffold, not a robotics disagreement protocol benchmark. The hostile prior-work pressure is strong: ManipArena-style real-world manipulation evaluation, robust ML under data imperfections, failure-aware RL with self-recovery, probabilistic ensemble forecasting, WorldBench physical disambiguation, embodied foundation models, systematic-noise robustness, robot failure tests, and uncertain-model robust control already cover much of the obvious territory. The rebuild cannot claim novelty from "ensembles disagree" or "uncertainty is useful." It must show that an action protocol turns disagreement into better robot decisions than ensemble uncertainty, conformal/risk filters, robust control, failure-aware recovery, and diagnostic probing.

## Rebuilt Claim Under Test

The strongest defensible claim is:

> Disagreement is useful only when the robot can classify disagreement type and choose an action protocol: commit, probe, switch controller, abstain, or recover.

This is a local evidence audit, not hardware validation.

## Benchmark Design

I will replace the template scaffold with a deterministic disagreement-to-action benchmark. Each episode samples a manipulation or locomotion scene, a hidden failure mode, an ensemble of noisy world/policy predictions, and a cost-sensitive action set. Methods decide whether disagreement means epistemic uncertainty, aleatoric noise, model misspecification, sensor corruption, out-of-support action, or harmless multimodality, then select an action protocol.

Tasks:

1. `bimanual_manipulation_failure_detection`
2. `contact_rich_pick_and_place`
3. `deformable_object_handling`
4. `mobile_navigation_under_sensor_noise`
5. `legged_locomotion_disturbance`

Disagreement families:

1. `epistemic_unknown`
2. `aleatoric_sensor_noise`
3. `model_misspecification`
4. `out_of_distribution_action`
5. `multi_modal_valid_plans`
6. `sensor_corruption`
7. `controller_instability`

Splits:

1. `nominal_disagreement`
2. `sensor_noise_shift`
3. `ood_action_shift`
4. `model_misspecification_shift`
5. `combined_disagreement_stress`

## Methods To Compare

Strong baselines:

1. `mean_ensemble_policy`
2. `variance_threshold_abstention`
3. `conformal_risk_filter`
4. `robust_mpc_fallback`
5. `failure_aware_rl_recovery`
6. `worldbench_diagnostic_probe`
7. `ensemble_forecasting_calibrator`
8. `proposed_disagreement_protocol`
9. `oracle_disagreement_type`

## Metrics

Disagreement metrics:

1. Disagreement-family classification accuracy.
2. Useful-disagreement recall.
3. Noise false-alarm rate.
4. Calibration error.
5. Probe informativeness.

Closed-loop metrics:

1. Task success.
2. Unsafe commit rate.
3. Unnecessary abstention/recovery rate.
4. Recovery success.
5. Protocol cost.
6. Planning regret to oracle.

Statistics:

1. Seven deterministic seeds.
2. Per-task and per-disagreement-family means with 95 percent confidence intervals.
3. Paired seed/task/family comparison against the strongest non-oracle baseline.
4. Explicit terminal decision in `results/summary.txt`.

## Ablations

The full method must beat stripped variants:

1. `full_disagreement_protocol`
2. `minus_disagreement_type_classifier`
3. `minus_probe_action`
4. `minus_protocol_cost_model`
5. `minus_calibration`
6. `variance_only_protocol`
7. `recovery_only_protocol`

If stripped variants match or beat full on task success, unsafe commits, or regret, the mechanism is not supported.

## Stress Tests

Stress axes:

1. Sensor noise.
2. Out-of-distribution action proposals.
3. Model misspecification.
4. Multimodal-valid action ambiguity.
5. Recovery/probe cost.
6. Combined maximum stress.

The stress sweep must show whether disagreement typing remains useful when variance thresholds, conformal filters, robust MPC, failure-aware recovery, diagnostic probes, and ensemble calibration degrade.

## Paper Rewrite Requirements

After experiments:

1. Rewrite `paper/main.tex` as either a strong-revise evidence report or a negative evidence audit.
2. Replace template claims with measured claims only.
3. Include tables for combined stress, ablations, pairwise decision, and failure cases.
4. Include figures for disagreement classification, closed-loop outcomes, cost/regret, ablations, and stress curves.
5. Update README, child status, claims, final audit, and submission-readiness docs.
6. Build only `C:/Users/wangz/Downloads/97.pdf`; do not copy anything to Desktop.
7. Commit and push to `https://github.com/Jason-Wang313/97_model_disagreement_protocols`.

## Terminal Gate

Mark `STRONG_REVISE` only if all of the following are true:

1. `proposed_disagreement_protocol` beats the strongest non-oracle baseline on combined-stress task success and unsafe-commit reduction.
2. It also improves useful-disagreement recall or disagreement-type accuracy without excessive false alarms, unnecessary abstentions, or protocol cost.
3. Core ablations degrade in expected directions.
4. Maximum-stress curves do not reverse in favor of conformal risk filters, robust MPC fallback, failure-aware RL recovery, diagnostic probes, or ensemble calibration.
5. The paper honestly states the evidence is local/simulated and not robot hardware validation.

Otherwise mark `KILL_ARCHIVE`. A disagreement protocol that is matched or beaten by variance thresholds, conformal filters, robust fallback, or failure-aware recovery is not ICLR-main ready.

## Execution Result

Executed on 2026-06-14. The rebuilt benchmark produced a terminal `KILL_ARCHIVE` decision. The proposed protocol improved useful-disagreement recall but did not beat `failure_aware_rl_recovery` on combined-stress task success (0.637 vs 0.701), did not improve over the safest baseline on unsafe commits, and was undermined by `minus_probe_action`, `minus_protocol_cost_model`, and `recovery_only_protocol` ablations that matched or beat full on success/regret.
