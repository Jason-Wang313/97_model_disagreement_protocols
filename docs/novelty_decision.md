# Novelty Decision

Chosen thesis: classify model disagreement and map it to robot action protocols.

Decision after v4/v4.1 evidence: KILL_ARCHIVE.

Reason: the claimed novelty is not empirically decisive. The audit shows better useful-disagreement recall, but the full mechanism loses to failure-aware recovery on task success and regret, and ablations remove parts of the proposed mechanism without hurting the measured objective.

Future revival would need evidence that the type-to-protocol mechanism, not merely recovery/fallback, causes robust robot performance gains.
