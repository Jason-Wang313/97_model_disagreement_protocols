# Hostile Reviewer Response

## Reviewer: A typed-disagreement taxonomy is unnecessary if direct recovery works.

Agreed. The v5 evidence supports this critique. `recovery_first_policy` and `failure_aware_rl_recovery` dominate the proposed method on the most important deployment metrics.

## Reviewer: Type accuracy is not enough.

Agreed. The manuscript explicitly separates type accuracy, useful recall, false alarms, utility, regret, and fixed-risk coverage. V5 lowers false alarms but fails deployment utility.

## Reviewer: The ablations do not isolate the mechanism.

Agreed. `no_fixed_risk_budget`, `no_probe_action`, `no_value_of_information_gate`, `recovery_only_protocol`, and `v4_protocol_rules` match or beat the full v5 mechanism on predefined metrics.

## Reviewer: The evidence is not robotics-real enough for ICLR main.

Agreed. The scope gate fails because there is no real robot, accepted high-fidelity simulator, or trained checkpoint evidence.

## Honest Response

Do not present this as a positive submission. The correct artifact is a negative archive that documents the failure mode and gives exact revival conditions.
