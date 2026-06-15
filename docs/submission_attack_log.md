# Submission Attack Log

Paper: 97 `model_disagreement_protocols`

Version: v4.1 rerun audit

## Attack 1: This is just uncertainty thresholding.

Result: Partially answered. The proposed protocol has better type accuracy and useful-disagreement recall than variance and many risk filters.

## Attack 2: The taxonomy does not improve closed-loop success.

Result: Fatal. On combined stress, `failure_aware_rl_recovery` reaches 0.701 task success while the proposed protocol reaches 0.637.

## Attack 3: Robust fallback already handles dangerous disagreement.

Result: Serious. `robust_mpc_fallback` is statistically tied or better on key safety/regret comparisons despite lower type accuracy.

## Attack 4: Probing costs more than it helps.

Result: Serious. Failure cases show epistemic and sensor-corruption scenarios where probe cost creates high regret. The `minus_probe_action` ablation matches or beats the full method on success/regret.

## Attack 5: The cost model is doing harm.

Result: Fatal. `minus_protocol_cost_model` matches or beats the full method on success/regret.

## Attack 6: Recovery alone is enough.

Result: Fatal. `recovery_only_protocol` beats the full protocol on combined-stress success and regret.

## Attack 7: The paper lacks real robot or high-fidelity validation.

Result: Still true. This prevents main-conference readiness even if the local result had been positive.

## Terminal Action

Mark `KILL_ARCHIVE`. Do not submit as an ICLR-main paper.

## Continuation Rerun Check

Result: unchanged. The 2026-06-15 rerun again finds that the proposed protocol loses combined-stress success to `failure_aware_rl_recovery` (`0.63692` vs `0.70093`), loses regret (`0.11064` vs `0.05130`), and is undercut by `minus_probe_action`, `minus_protocol_cost_model`, and `recovery_only_protocol`.
