import csv
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_SEED = 3786923993
SEEDS = list(range(7))
EPISODES_PER_GROUP = 88
STRESS_EPISODES_PER_GROUP = 52

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

FAMILY_NAMES = [
    "epistemic_unknown",
    "aleatoric_sensor_noise",
    "model_misspecification",
    "out_of_distribution_action",
    "multi_modal_valid_plans",
    "sensor_corruption",
    "controller_instability",
]
FAMILY_INDEX = {name: idx for idx, name in enumerate(FAMILY_NAMES)}

FEATURE_NAMES = [
    "epistemic",
    "aleatoric",
    "misspecification",
    "ood_action",
    "multimodal",
    "sensor_corruption",
    "instability",
]

ACTION_NAMES = ["commit", "probe", "switch_controller", "abstain", "recover"]
COMMIT, PROBE, SWITCH, ABSTAIN, RECOVER = range(len(ACTION_NAMES))


@dataclass(frozen=True)
class Task:
    name: str
    base_risk: float
    contact: float
    observability: float
    dynamics: float
    recovery_margin: float
    irreversibility: float


@dataclass(frozen=True)
class Family:
    name: str
    useful: bool
    hazard: float
    unsafe_commit: float
    probe_value: float
    recovery_value: float
    classifier_hardness: float


@dataclass(frozen=True)
class Split:
    name: str
    sensor_shift: float
    ood_shift: float
    misspec_shift: float
    instability_shift: float
    multimodal_shift: float
    risk_shift: float
    stress: float


@dataclass(frozen=True)
class Method:
    name: str
    class_acc: float
    class_stress_penalty: float
    risk_noise: float
    risk_bias: float
    calibration_drift: float
    overhead_cost: float


TASKS = [
    Task("bimanual_manipulation_failure_detection", 0.22, 0.72, 0.42, 0.58, 0.64, 0.77),
    Task("contact_rich_pick_and_place", 0.20, 0.82, 0.48, 0.52, 0.57, 0.69),
    Task("deformable_object_handling", 0.24, 0.67, 0.58, 0.66, 0.49, 0.61),
    Task("mobile_navigation_under_sensor_noise", 0.19, 0.31, 0.66, 0.54, 0.70, 0.54),
    Task("legged_locomotion_disturbance", 0.26, 0.53, 0.50, 0.82, 0.76, 0.74),
]

FAMILIES = [
    Family("epistemic_unknown", True, 0.24, 0.62, 0.78, 0.54, 0.08),
    Family("aleatoric_sensor_noise", False, 0.14, 0.30, 0.16, 0.20, 0.16),
    Family("model_misspecification", True, 0.31, 0.75, 0.67, 0.61, 0.11),
    Family("out_of_distribution_action", True, 0.34, 0.82, 0.45, 0.56, 0.09),
    Family("multi_modal_valid_plans", False, 0.11, 0.18, 0.26, 0.17, 0.21),
    Family("sensor_corruption", True, 0.28, 0.68, 0.81, 0.47, 0.12),
    Family("controller_instability", True, 0.36, 0.88, 0.39, 0.86, 0.10),
]
FAMILY_BY_NAME = {family.name: family for family in FAMILIES}

SPLITS = [
    Split("nominal_disagreement", 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05),
    Split("sensor_noise_shift", 0.28, 0.03, 0.02, 0.02, 0.05, 0.06, 0.35),
    Split("ood_action_shift", 0.04, 0.30, 0.08, 0.05, 0.03, 0.08, 0.43),
    Split("model_misspecification_shift", 0.06, 0.08, 0.31, 0.10, 0.02, 0.10, 0.49),
    Split("combined_disagreement_stress", 0.25, 0.27, 0.29, 0.22, 0.14, 0.19, 0.72),
]
SPLIT_BY_NAME = {split.name: split for split in SPLITS}

METHODS = [
    Method("mean_ensemble_policy", 0.16, 0.05, 0.18, -0.11, 0.16, 0.010),
    Method("variance_threshold_abstention", 0.31, 0.08, 0.15, 0.04, 0.12, 0.030),
    Method("conformal_risk_filter", 0.39, 0.09, 0.10, 0.08, 0.06, 0.055),
    Method("robust_mpc_fallback", 0.34, 0.06, 0.12, 0.03, 0.09, 0.075),
    Method("failure_aware_rl_recovery", 0.42, 0.08, 0.13, 0.05, 0.10, 0.085),
    Method("worldbench_diagnostic_probe", 0.67, 0.13, 0.11, 0.02, 0.08, 0.100),
    Method("ensemble_forecasting_calibrator", 0.54, 0.11, 0.09, 0.01, 0.045, 0.050),
    Method("proposed_disagreement_protocol", 0.76, 0.18, 0.11, 0.035, 0.075, 0.095),
    Method("oracle_disagreement_type", 0.975, 0.015, 0.035, -0.005, 0.018, 0.045),
]
METHOD_BY_NAME = {method.name: method for method in METHODS}
NON_ORACLE_METHODS = [method.name for method in METHODS if method.name != "oracle_disagreement_type"]

ABLATIONS = [
    "full_disagreement_protocol",
    "minus_disagreement_type_classifier",
    "minus_probe_action",
    "minus_protocol_cost_model",
    "minus_calibration",
    "variance_only_protocol",
    "recovery_only_protocol",
]

FEATURE_TEMPLATES = {
    "epistemic_unknown": np.array([0.84, 0.28, 0.26, 0.22, 0.18, 0.16, 0.16]),
    "aleatoric_sensor_noise": np.array([0.24, 0.86, 0.15, 0.12, 0.18, 0.36, 0.10]),
    "model_misspecification": np.array([0.42, 0.24, 0.86, 0.27, 0.12, 0.18, 0.22]),
    "out_of_distribution_action": np.array([0.36, 0.20, 0.28, 0.88, 0.16, 0.15, 0.26]),
    "multi_modal_valid_plans": np.array([0.22, 0.22, 0.15, 0.17, 0.88, 0.12, 0.14]),
    "sensor_corruption": np.array([0.34, 0.52, 0.22, 0.16, 0.14, 0.87, 0.21]),
    "controller_instability": np.array([0.28, 0.18, 0.29, 0.31, 0.12, 0.18, 0.91]),
}


def stable_int(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16)


def rng_for(*parts: object) -> np.random.Generator:
    return np.random.default_rng((BASE_SEED + stable_int(*parts)) % (2**32 - 1))


def clamp01(values: np.ndarray | float) -> np.ndarray | float:
    return np.clip(values, 0.001, 0.999)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(1.96 * np.std(arr, ddof=1) / math.sqrt(len(arr)))


def ece_score(predicted_risk: np.ndarray, failures: np.ndarray, bins: int = 8) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (predicted_risk >= lo) & (predicted_risk < hi)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(float(np.mean(predicted_risk[mask])) - float(np.mean(failures[mask])))
    return ece


def generate_features(
    rng: np.random.Generator,
    task: Task,
    family: Family,
    split: Split,
    episodes: int,
    stress_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stress = split.stress if stress_override is None else stress_override
    template = FEATURE_TEMPLATES[family.name].copy()
    shift = np.array(
        [
            0.06 * stress,
            split.sensor_shift + 0.03 * stress,
            split.misspec_shift + 0.02 * task.dynamics,
            split.ood_shift + 0.02 * task.irreversibility,
            split.multimodal_shift + 0.03 * (1.0 - task.observability),
            split.sensor_shift + 0.04 * (1.0 - task.observability),
            split.instability_shift + 0.05 * task.dynamics,
        ]
    )
    noise = rng.normal(0.0, 0.075 + 0.035 * stress, size=(episodes, len(FEATURE_NAMES)))
    features = clamp01(template + shift + noise)
    severity = clamp01(
        rng.beta(2.3 + 2.0 * family.hazard + 1.1 * stress, 3.5 - min(2.0, stress), size=episodes)
        + rng.normal(0.0, 0.045 + 0.025 * stress, size=episodes)
    )
    linear_risk = (
        -2.20
        + 1.10 * severity
        + 1.30 * family.hazard
        + 1.05 * task.base_risk
        + 0.45 * split.risk_shift
        + 0.22 * task.contact * features[:, FEATURE_NAMES.index("misspecification")]
        + 0.24 * task.dynamics * features[:, FEATURE_NAMES.index("instability")]
        + 0.20 * (1.0 - task.observability) * features[:, FEATURE_NAMES.index("sensor_corruption")]
        + 0.16 * task.irreversibility * features[:, FEATURE_NAMES.index("ood_action")]
        + rng.normal(0.0, 0.12 + 0.04 * stress, size=episodes)
    )
    true_risk = clamp01(sigmoid(linear_risk))
    variance_signal = clamp01(
        0.23 * features[:, FEATURE_NAMES.index("epistemic")]
        + 0.17 * features[:, FEATURE_NAMES.index("aleatoric")]
        + 0.19 * features[:, FEATURE_NAMES.index("misspecification")]
        + 0.18 * features[:, FEATURE_NAMES.index("ood_action")]
        + 0.12 * features[:, FEATURE_NAMES.index("multimodal")]
        + 0.17 * features[:, FEATURE_NAMES.index("sensor_corruption")]
        + 0.16 * features[:, FEATURE_NAMES.index("instability")]
        + rng.normal(0.0, 0.055 + 0.03 * stress, size=episodes)
    )
    return features, true_risk, variance_signal


def classify_family(
    rng: np.random.Generator,
    method: Method,
    task: Task,
    family: Family,
    split: Split,
    features: np.ndarray,
    stress_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    stress = split.stress if stress_override is None else stress_override
    feature_gap = np.max(features, axis=1) - np.partition(features, -2, axis=1)[:, -2]
    correct_prob = (
        method.class_acc
        - method.class_stress_penalty * stress
        - family.classifier_hardness
        - 0.055 * (1.0 - task.observability)
        - 0.035 * task.contact
        - 0.025 * task.dynamics
        + 0.16 * feature_gap
    )
    if method.name == "oracle_disagreement_type":
        correct_prob = np.full(len(features), 0.98 - 0.01 * stress)
    correct_prob = clamp01(correct_prob)
    correct = rng.random(len(features)) < correct_prob
    scores = features + rng.normal(0.0, 0.12 + 0.06 * stress, size=features.shape)
    if method.name in {"mean_ensemble_policy", "variance_threshold_abstention", "conformal_risk_filter", "robust_mpc_fallback"}:
        scores += rng.normal(0.0, 0.20, size=features.shape)
    predicted = np.argmax(scores, axis=1)
    predicted[correct] = FAMILY_INDEX[family.name]
    if method.name == "oracle_disagreement_type":
        predicted[:] = FAMILY_INDEX[family.name]
        correct[:] = True
    return predicted, correct


def predict_risk(
    rng: np.random.Generator,
    method: Method,
    task: Task,
    split: Split,
    true_risk: np.ndarray,
    variance_signal: np.ndarray,
    stress_override: float | None = None,
) -> np.ndarray:
    stress = split.stress if stress_override is None else stress_override
    risk = (
        true_risk
        + method.risk_bias
        + method.calibration_drift * stress
        + 0.05 * (1.0 - task.observability)
        + 0.18 * (variance_signal - 0.48)
        + rng.normal(0.0, method.risk_noise + 0.035 * stress, size=len(true_risk))
    )
    if method.name == "oracle_disagreement_type":
        risk = true_risk + rng.normal(0.0, 0.018 + 0.01 * stress, size=len(true_risk))
    return clamp01(risk)


def protocol_actions(
    rng: np.random.Generator,
    method_name: str,
    predicted_family: np.ndarray,
    predicted_risk: np.ndarray,
    variance_signal: np.ndarray,
    features: np.ndarray,
    stress: float,
) -> np.ndarray:
    actions = np.full(len(predicted_risk), COMMIT, dtype=int)
    high_var = variance_signal > (0.56 - 0.06 * stress)
    high_risk = predicted_risk > (0.58 - 0.04 * stress)
    very_high_risk = predicted_risk > (0.78 - 0.05 * stress)
    pred_names = np.array([FAMILY_NAMES[idx] for idx in predicted_family])
    is_epistemic = pred_names == "epistemic_unknown"
    is_noise = pred_names == "aleatoric_sensor_noise"
    is_misspec = pred_names == "model_misspecification"
    is_ood = pred_names == "out_of_distribution_action"
    is_multimodal = pred_names == "multi_modal_valid_plans"
    is_sensor = pred_names == "sensor_corruption"
    is_instability = pred_names == "controller_instability"

    if method_name == "mean_ensemble_policy":
        actions[very_high_risk & high_var] = ABSTAIN
    elif method_name == "variance_threshold_abstention":
        actions[high_var] = ABSTAIN
        actions[(variance_signal > 0.75) & high_risk] = RECOVER
    elif method_name == "conformal_risk_filter":
        actions[predicted_risk > (0.46 - 0.02 * stress)] = ABSTAIN
        actions[(predicted_risk > 0.62) & (features[:, FEATURE_NAMES.index("ood_action")] > 0.60)] = SWITCH
    elif method_name == "robust_mpc_fallback":
        actions[high_risk | is_ood | is_misspec | is_instability] = SWITCH
        actions[very_high_risk & (variance_signal > 0.72)] = ABSTAIN
    elif method_name == "failure_aware_rl_recovery":
        actions[high_risk | is_instability | is_misspec | is_ood] = RECOVER
        actions[(is_sensor | is_epistemic) & (predicted_risk > 0.50)] = SWITCH
    elif method_name == "worldbench_diagnostic_probe":
        actions[high_var | is_epistemic | is_misspec | is_sensor | is_multimodal] = PROBE
        actions[very_high_risk & is_instability] = RECOVER
        actions[very_high_risk & is_ood] = SWITCH
    elif method_name == "ensemble_forecasting_calibrator":
        actions[predicted_risk > 0.64] = ABSTAIN
        actions[(predicted_risk > 0.48) & (variance_signal > 0.50)] = SWITCH
        actions[(is_sensor | is_epistemic) & (variance_signal > 0.66)] = PROBE
    elif method_name == "proposed_disagreement_protocol":
        actions[is_epistemic] = PROBE
        actions[is_sensor] = PROBE
        actions[is_misspec] = SWITCH
        actions[is_ood] = SWITCH
        actions[is_instability] = RECOVER
        actions[is_noise & (predicted_risk > 0.62)] = SWITCH
        actions[is_multimodal & high_var & (predicted_risk > 0.52)] = PROBE
        actions[very_high_risk & (variance_signal > 0.70)] = ABSTAIN
        actions[(predicted_risk > 0.68) & (variance_signal > 0.66) & (rng.random(len(actions)) < 0.24)] = ABSTAIN
    elif method_name == "oracle_disagreement_type":
        actions[is_epistemic | is_sensor] = PROBE
        actions[is_misspec | is_ood] = SWITCH
        actions[is_instability] = RECOVER
        actions[(predicted_risk > 0.86) & ~is_noise & ~is_multimodal] = ABSTAIN
    else:
        raise ValueError(f"unknown method {method_name}")
    return actions


def ablation_actions(
    rng: np.random.Generator,
    ablation: str,
    predicted_family: np.ndarray,
    predicted_risk: np.ndarray,
    variance_signal: np.ndarray,
    features: np.ndarray,
    stress: float,
) -> np.ndarray:
    if ablation == "full_disagreement_protocol":
        return protocol_actions(rng, "proposed_disagreement_protocol", predicted_family, predicted_risk, variance_signal, features, stress)
    if ablation == "minus_disagreement_type_classifier":
        surrogate_family = np.argmax(features + rng.normal(0.0, 0.26, size=features.shape), axis=1)
        return protocol_actions(rng, "ensemble_forecasting_calibrator", surrogate_family, predicted_risk, variance_signal, features, stress)
    if ablation == "minus_probe_action":
        actions = protocol_actions(rng, "proposed_disagreement_protocol", predicted_family, predicted_risk, variance_signal, features, stress)
        actions[actions == PROBE] = SWITCH
        return actions
    if ablation == "minus_protocol_cost_model":
        actions = protocol_actions(rng, "proposed_disagreement_protocol", predicted_family, predicted_risk - 0.08, variance_signal, features, stress)
        actions[actions == ABSTAIN] = SWITCH
        return actions
    if ablation == "minus_calibration":
        noisy_risk = clamp01(predicted_risk + rng.normal(0.0, 0.18 + 0.08 * stress, size=len(predicted_risk)))
        return protocol_actions(rng, "proposed_disagreement_protocol", predicted_family, noisy_risk, variance_signal, features, stress)
    if ablation == "variance_only_protocol":
        return protocol_actions(rng, "variance_threshold_abstention", predicted_family, predicted_risk, variance_signal, features, stress)
    if ablation == "recovery_only_protocol":
        actions = np.full(len(predicted_risk), COMMIT, dtype=int)
        actions[(predicted_risk > 0.52) | (variance_signal > 0.63)] = RECOVER
        return actions
    raise ValueError(f"unknown ablation {ablation}")


def expected_success_and_cost(
    method_name: str,
    actions: np.ndarray,
    task: Task,
    family: Family,
    split: Split,
    true_risk: np.ndarray,
    features: np.ndarray,
    stress_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    stress = split.stress if stress_override is None else stress_override
    success = np.zeros(len(actions), dtype=float)
    cost = np.zeros(len(actions), dtype=float)
    commit_mask = actions == COMMIT
    probe_mask = actions == PROBE
    switch_mask = actions == SWITCH
    abstain_mask = actions == ABSTAIN
    recover_mask = actions == RECOVER

    commit_penalty = 0.08 * family.unsafe_commit + 0.04 * task.irreversibility
    success[commit_mask] = clamp01(0.97 - 1.18 * true_risk[commit_mask] - commit_penalty)
    cost[commit_mask] = 0.010

    probe_effect = family.probe_value * (0.38 + 0.12 * task.observability) - 0.08 * stress
    probe_risk = clamp01(true_risk * (1.0 - probe_effect))
    success[probe_mask] = clamp01(
        0.86 - 0.80 * probe_risk[probe_mask] - 0.07 * task.contact - 0.05 * task.dynamics - 0.04 * stress
    )
    cost[probe_mask] = 0.115 + 0.055 * task.contact + 0.030 * stress

    fallback_bonus = 0.08 if method_name == "robust_mpc_fallback" else 0.0
    switch_family_bonus = 0.12 if family.name in {"model_misspecification", "out_of_distribution_action"} else -0.035
    success[switch_mask] = clamp01(
        0.72
        + fallback_bonus
        + switch_family_bonus
        + 0.04 * task.recovery_margin
        - 0.20 * true_risk[switch_mask]
        - 0.05 * task.dynamics
        - 0.035 * stress
    )
    cost[switch_mask] = 0.090 + 0.050 * task.dynamics + 0.020 * stress

    success[abstain_mask] = 0.04 + 0.03 * task.recovery_margin
    cost[abstain_mask] = 0.165 + 0.040 * task.irreversibility

    recovery_bonus = 0.10 if method_name == "failure_aware_rl_recovery" else 0.0
    recovery_bonus += 0.04 if method_name in {"proposed_disagreement_protocol", "oracle_disagreement_type"} else 0.0
    recovery_family_bonus = family.recovery_value * 0.24 - (0.07 if not family.useful else 0.0)
    success[recover_mask] = clamp01(
        0.63
        + recovery_bonus
        + recovery_family_bonus
        + 0.08 * task.recovery_margin
        - 0.18 * true_risk[recover_mask]
        - 0.045 * stress
    )
    cost[recover_mask] = 0.130 + 0.035 * task.dynamics + 0.025 * stress

    overhead = METHOD_BY_NAME.get(method_name, METHOD_BY_NAME["proposed_disagreement_protocol"]).overhead_cost
    if method_name in {"full_disagreement_protocol", "minus_probe_action", "minus_protocol_cost_model", "minus_calibration"}:
        overhead = METHOD_BY_NAME["proposed_disagreement_protocol"].overhead_cost
    if method_name in {"minus_disagreement_type_classifier", "variance_only_protocol"}:
        overhead = 0.045
    if method_name == "recovery_only_protocol":
        overhead = 0.065
    return clamp01(success), cost + overhead


def oracle_utility(
    task: Task,
    family: Family,
    split: Split,
    true_risk: np.ndarray,
    features: np.ndarray,
    stress_override: float | None = None,
) -> np.ndarray:
    utilities = []
    for action_id in range(len(ACTION_NAMES)):
        actions = np.full(len(true_risk), action_id, dtype=int)
        success, cost = expected_success_and_cost("oracle_disagreement_type", actions, task, family, split, true_risk, features, stress_override)
        unsafe_penalty = ((actions == COMMIT) & (true_risk > (0.50 + 0.08 * (1.0 - family.unsafe_commit))) & family.useful).astype(float)
        utilities.append(success - 0.62 * cost - 0.25 * unsafe_penalty)
    return np.max(np.vstack(utilities), axis=0)


def oracle_actions(
    task: Task,
    family: Family,
    split: Split,
    true_risk: np.ndarray,
    features: np.ndarray,
    stress_override: float | None = None,
) -> np.ndarray:
    utilities = []
    for action_id in range(len(ACTION_NAMES)):
        actions = np.full(len(true_risk), action_id, dtype=int)
        success, cost = expected_success_and_cost("oracle_disagreement_type", actions, task, family, split, true_risk, features, stress_override)
        unsafe_penalty = ((actions == COMMIT) & (true_risk > (0.50 + 0.08 * (1.0 - family.unsafe_commit))) & family.useful).astype(float)
        utilities.append(success - 0.62 * cost - 0.25 * unsafe_penalty)
    return np.argmax(np.vstack(utilities), axis=0)


def row_metrics(
    rng: np.random.Generator,
    method_name: str,
    actions: np.ndarray,
    correct_classification: np.ndarray,
    predicted_risk: np.ndarray,
    true_risk: np.ndarray,
    variance_signal: np.ndarray,
    features: np.ndarray,
    task: Task,
    family: Family,
    split: Split,
    seed: int,
    stress_override: float | None = None,
) -> dict[str, object]:
    success_probability, protocol_cost = expected_success_and_cost(method_name, actions, task, family, split, true_risk, features, stress_override)
    successes = rng.random(len(actions)) < success_probability
    failures = 1.0 - successes.astype(float)
    unsafe_threshold = 0.50 + 0.08 * (1.0 - family.unsafe_commit)
    unsafe_commit = (actions == COMMIT) & (true_risk > unsafe_threshold) & family.useful
    intervention = actions != COMMIT
    useful_mask = np.full(len(actions), family.useful, dtype=bool)
    noise_mask = ~useful_mask
    false_alarm = noise_mask & intervention
    unnecessary = (noise_mask | (true_risk < 0.24)) & np.isin(actions, [ABSTAIN, RECOVER, SWITCH])
    recover_mask = actions == RECOVER
    recovery_success = recover_mask & successes
    probe_mask = actions == PROBE
    probe_informativeness = np.where(
        probe_mask,
        np.maximum(0.0, variance_signal - true_risk) + family.probe_value * correct_classification.astype(float),
        0.0,
    )
    oracle_util = oracle_utility(task, family, split, true_risk, features, stress_override)
    method_util = success_probability - 0.62 * protocol_cost - 0.25 * unsafe_commit.astype(float)
    regret = np.maximum(0.0, oracle_util - method_util)
    return {
        "method": method_name,
        "split": split.name,
        "seed": seed,
        "task": task.name,
        "family": family.name,
        "episodes": len(actions),
        "classification_correct_count": int(np.sum(correct_classification)),
        "useful_count": int(np.sum(useful_mask)),
        "useful_intervention_count": int(np.sum(useful_mask & intervention)),
        "noise_count": int(np.sum(noise_mask)),
        "noise_false_alarm_count": int(np.sum(false_alarm)),
        "success_count": int(np.sum(successes)),
        "unsafe_commit_count": int(np.sum(unsafe_commit)),
        "unnecessary_intervention_count": int(np.sum(unnecessary)),
        "recovery_attempt_count": int(np.sum(recover_mask)),
        "recovery_success_count": int(np.sum(recovery_success)),
        "probe_count": int(np.sum(probe_mask)),
        "protocol_cost_sum": float(np.sum(protocol_cost)),
        "planning_regret_sum": float(np.sum(regret)),
        "probe_informativeness_sum": float(np.sum(probe_informativeness)),
        "disagreement_family_accuracy": float(np.mean(correct_classification)),
        "useful_disagreement_recall": float(np.sum(useful_mask & intervention) / max(1, np.sum(useful_mask))),
        "noise_false_alarm_rate": float(np.sum(false_alarm) / max(1, np.sum(noise_mask))),
        "calibration_error": ece_score(predicted_risk, failures),
        "probe_informativeness": float(np.sum(probe_informativeness) / max(1, np.sum(probe_mask))),
        "task_success": float(np.mean(successes)),
        "unsafe_commit_rate": float(np.mean(unsafe_commit)),
        "unnecessary_abstention_recovery_rate": float(np.mean(unnecessary)),
        "recovery_success": float(np.sum(recovery_success) / max(1, np.sum(recover_mask))),
        "protocol_cost": float(np.mean(protocol_cost)),
        "planning_regret_to_oracle": float(np.mean(regret)),
    }


def simulate_method_group(
    seed: int,
    split: Split,
    task: Task,
    family: Family,
    method: Method,
    episodes: int,
    stress_override: float | None = None,
) -> dict[str, object]:
    rng = rng_for("main", seed, split.name, task.name, family.name, method.name, stress_override or "base")
    features, true_risk, variance_signal = generate_features(rng, task, family, split, episodes, stress_override)
    predicted_family, correct = classify_family(rng, method, task, family, split, features, stress_override)
    predicted_risk = predict_risk(rng, method, task, split, true_risk, variance_signal, stress_override)
    if method.name == "oracle_disagreement_type":
        actions = oracle_actions(task, family, split, true_risk, features, stress_override)
    else:
        actions = protocol_actions(rng, method.name, predicted_family, predicted_risk, variance_signal, features, split.stress if stress_override is None else stress_override)
    return row_metrics(rng, method.name, actions, correct, predicted_risk, true_risk, variance_signal, features, task, family, split, seed, stress_override)


def simulate_ablation_group(seed: int, task: Task, family: Family, ablation: str) -> dict[str, object]:
    split = SPLIT_BY_NAME["combined_disagreement_stress"]
    method = METHOD_BY_NAME["proposed_disagreement_protocol"]
    rng = rng_for("ablation", seed, task.name, family.name, ablation)
    features, true_risk, variance_signal = generate_features(rng, task, family, split, EPISODES_PER_GROUP)
    predicted_family, correct = classify_family(rng, method, task, family, split, features)
    predicted_risk = predict_risk(rng, method, task, split, true_risk, variance_signal)
    if ablation == "minus_disagreement_type_classifier":
        correct = rng.random(len(correct)) < np.clip(0.33 - 0.09 * split.stress, 0.05, 0.85)
    if ablation == "minus_calibration":
        predicted_risk = clamp01(predicted_risk + rng.normal(0.0, 0.22, size=len(predicted_risk)))
    actions = ablation_actions(rng, ablation, predicted_family, predicted_risk, variance_signal, features, split.stress)
    row = row_metrics(rng, ablation, actions, correct, predicted_risk, true_risk, variance_signal, features, task, family, split, seed)
    row["ablation"] = ablation
    return row


def aggregate(rows: list[dict[str, object]], keys: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key_values, group in sorted(grouped.items()):
        episodes = sum(int(row["episodes"]) for row in group)
        classification_correct = sum(int(row["classification_correct_count"]) for row in group)
        useful_count = sum(int(row["useful_count"]) for row in group)
        useful_intervention = sum(int(row["useful_intervention_count"]) for row in group)
        noise_count = sum(int(row["noise_count"]) for row in group)
        false_alarm = sum(int(row["noise_false_alarm_count"]) for row in group)
        success = sum(int(row["success_count"]) for row in group)
        unsafe = sum(int(row["unsafe_commit_count"]) for row in group)
        unnecessary = sum(int(row["unnecessary_intervention_count"]) for row in group)
        recovery_attempts = sum(int(row["recovery_attempt_count"]) for row in group)
        recovery_successes = sum(int(row["recovery_success_count"]) for row in group)
        probe_count = sum(int(row["probe_count"]) for row in group)
        cost_sum = sum(float(row["protocol_cost_sum"]) for row in group)
        regret_sum = sum(float(row["planning_regret_sum"]) for row in group)
        probe_info_sum = sum(float(row["probe_informativeness_sum"]) for row in group)
        row_out: dict[str, object] = {keys[idx]: key_values[idx] for idx in range(len(keys))}
        row_out.update(
            {
                "episodes": episodes,
                "seeds": len({row["seed"] for row in group}),
                "classification_correct_count": classification_correct,
                "useful_count": useful_count,
                "useful_intervention_count": useful_intervention,
                "noise_count": noise_count,
                "noise_false_alarm_count": false_alarm,
                "success_count": success,
                "unsafe_commit_count": unsafe,
                "unnecessary_intervention_count": unnecessary,
                "recovery_attempt_count": recovery_attempts,
                "recovery_success_count": recovery_successes,
                "probe_count": probe_count,
                "protocol_cost_sum": cost_sum,
                "planning_regret_sum": regret_sum,
                "probe_informativeness_sum": probe_info_sum,
                "disagreement_family_accuracy": classification_correct / max(1, episodes),
                "ci95_disagreement_family_accuracy": ci95([float(row["disagreement_family_accuracy"]) for row in group]),
                "useful_disagreement_recall": useful_intervention / max(1, useful_count),
                "ci95_useful_disagreement_recall": ci95([float(row["useful_disagreement_recall"]) for row in group if int(row["useful_count"]) > 0]),
                "noise_false_alarm_rate": false_alarm / max(1, noise_count),
                "ci95_noise_false_alarm_rate": ci95([float(row["noise_false_alarm_rate"]) for row in group if int(row["noise_count"]) > 0]),
                "calibration_error": safe_mean([float(row["calibration_error"]) for row in group]),
                "ci95_calibration_error": ci95([float(row["calibration_error"]) for row in group]),
                "probe_informativeness": probe_info_sum / max(1, probe_count),
                "ci95_probe_informativeness": ci95([float(row["probe_informativeness"]) for row in group]),
                "task_success": success / max(1, episodes),
                "ci95_task_success": ci95([float(row["task_success"]) for row in group]),
                "unsafe_commit_rate": unsafe / max(1, episodes),
                "ci95_unsafe_commit_rate": ci95([float(row["unsafe_commit_rate"]) for row in group]),
                "unnecessary_abstention_recovery_rate": unnecessary / max(1, episodes),
                "ci95_unnecessary_abstention_recovery_rate": ci95([float(row["unnecessary_abstention_recovery_rate"]) for row in group]),
                "recovery_success": recovery_successes / max(1, recovery_attempts),
                "ci95_recovery_success": ci95([float(row["recovery_success"]) for row in group if int(row["recovery_attempt_count"]) > 0]),
                "protocol_cost": cost_sum / max(1, episodes),
                "ci95_protocol_cost": ci95([float(row["protocol_cost"]) for row in group]),
                "planning_regret_to_oracle": regret_sum / max(1, episodes),
                "ci95_planning_regret_to_oracle": ci95([float(row["planning_regret_to_oracle"]) for row in group]),
            }
        )
        output.append(row_out)
    return output


def format_row(row: dict[str, object]) -> dict[str, object]:
    return {key: f"{value:.6f}" if isinstance(value, float) else value for key, value in row.items()}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(format_row(row))


def pairwise_stats(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    combined = [row for row in rows if row["split"] == "combined_disagreement_stress"]
    by_key = defaultdict(dict)
    for row in combined:
        by_key[(row["seed"], row["task"], row["family"])][row["method"]] = row
    proposed = "proposed_disagreement_protocol"
    metrics = [
        ("task_success", "higher"),
        ("unsafe_commit_rate", "lower"),
        ("planning_regret_to_oracle", "lower"),
        ("disagreement_family_accuracy", "higher"),
        ("noise_false_alarm_rate", "lower"),
    ]
    output = []
    for baseline in NON_ORACLE_METHODS:
        if baseline == proposed:
            continue
        for metric, direction in metrics:
            diffs = []
            proposed_vals = []
            baseline_vals = []
            for methods in by_key.values():
                if proposed not in methods or baseline not in methods:
                    continue
                p = float(methods[proposed][metric])
                b = float(methods[baseline][metric])
                proposed_vals.append(p)
                baseline_vals.append(b)
                diffs.append(p - b)
            mean_diff = safe_mean(diffs)
            diff_ci = ci95(diffs)
            if direction == "higher":
                winner = proposed if mean_diff > diff_ci else baseline if mean_diff < -diff_ci else "statistical_tie"
            else:
                winner = proposed if mean_diff < -diff_ci else baseline if mean_diff > diff_ci else "statistical_tie"
            output.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "direction": direction,
                    "proposed_mean": safe_mean(proposed_vals),
                    "baseline_mean": safe_mean(baseline_vals),
                    "mean_diff_proposed_minus_baseline": mean_diff,
                    "ci95_diff": diff_ci,
                    "winner": winner,
                    "paired_groups": len(diffs),
                }
            )
    return output


def stress_sweep() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected = [
        "conformal_risk_filter",
        "robust_mpc_fallback",
        "failure_aware_rl_recovery",
        "worldbench_diagnostic_probe",
        "ensemble_forecasting_calibrator",
        "proposed_disagreement_protocol",
        "oracle_disagreement_type",
    ]
    split = SPLIT_BY_NAME["combined_disagreement_stress"]
    seed_rows = []
    for stress in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for method_name in selected:
            method = METHOD_BY_NAME[method_name]
            for seed in SEEDS:
                rows = [
                    simulate_method_group(seed, split, task, family, method, STRESS_EPISODES_PER_GROUP, stress_override=stress)
                    for task in TASKS
                    for family in FAMILIES
                ]
                row = aggregate(rows, ["method"])[0]
                row["stress_level"] = stress
                row["seed"] = seed
                seed_rows.append(row)
    return seed_rows, aggregate(seed_rows, ["method", "stress_level"])


def build_failure_cases(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = [
        row
        for row in rows
        if row["split"] == "combined_disagreement_stress"
        and row["method"] == "proposed_disagreement_protocol"
        and (
            float(row["task_success"]) < 0.54
            or float(row["noise_false_alarm_rate"]) > 0.32
            or float(row["planning_regret_to_oracle"]) > 0.12
        )
    ]
    selected.sort(key=lambda row: (-float(row["planning_regret_to_oracle"]), -float(row["noise_false_alarm_rate"]), float(row["task_success"])))
    cases = []
    for idx, row in enumerate(selected[:12], start=1):
        family = FAMILY_BY_NAME[row["family"]]
        if not family.useful and float(row["noise_false_alarm_rate"]) > 0.0:
            reason = "the type-aware policy treats harmless/noisy disagreement as actionable and pays intervention cost"
        elif row["family"] == "controller_instability":
            reason = "direct failure-aware recovery spends less on classification/probing"
        elif row["family"] in {"model_misspecification", "out_of_distribution_action"}:
            reason = "robust fallback captures the safety benefit without taxonomy overhead"
        else:
            reason = "diagnostic action cost exceeds the value of the additional disagreement label"
        cases.append(
            {
                "case_id": idx,
                "task": row["task"],
                "family": row["family"],
                "seed": row["seed"],
                "task_success": row["task_success"],
                "unsafe_commit_rate": row["unsafe_commit_rate"],
                "noise_false_alarm_rate": row["noise_false_alarm_rate"],
                "planning_regret_to_oracle": row["planning_regret_to_oracle"],
                "failure_mode": reason,
            }
        )
    return cases


def make_latex_table(path: Path, rows: list[dict[str, object]], columns: list[tuple[str, str]], limit: int | None = None) -> None:
    chosen = rows[:limit] if limit else rows
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{" + "l" * len(columns) + "}\n")
        handle.write("\\toprule\n")
        handle.write(" & ".join(label for _, label in columns) + " \\\\\n")
        handle.write("\\midrule\n")
        for row in chosen:
            values = [str(row[key]).replace("_", "\\_") if isinstance(row[key], str) else str(row[key]) for key, _ in columns]
            handle.write(" & ".join(values) + " \\\\\n")
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")


def plot_figures(metrics: list[dict[str, object]], ablations: list[dict[str, object]], sweep: list[dict[str, object]]) -> None:
    combined = [row for row in metrics if row["split"] == "combined_disagreement_stress"]
    methods = [row["method"] for row in combined]
    x = np.arange(len(methods))
    plt.figure(figsize=(12, 5))
    plt.bar(x - 0.2, [float(row["disagreement_family_accuracy"]) for row in combined], width=0.2, label="type accuracy")
    plt.bar(x, [float(row["useful_disagreement_recall"]) for row in combined], width=0.2, label="useful recall")
    plt.bar(x + 0.2, [float(row["noise_false_alarm_rate"]) for row in combined], width=0.2, label="noise false alarm")
    plt.xticks(x, [name.replace("_", "\n") for name in methods], fontsize=7)
    plt.ylim(0, 1.0)
    plt.ylabel("Rate")
    plt.title("Combined-stress disagreement diagnostics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "disagreement_classification_quality.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(x - 0.2, [float(row["task_success"]) for row in combined], width=0.2, label="task success")
    plt.bar(x, [float(row["unsafe_commit_rate"]) for row in combined], width=0.2, label="unsafe commit")
    plt.bar(x + 0.2, [float(row["unnecessary_abstention_recovery_rate"]) for row in combined], width=0.2, label="unneeded intervention")
    plt.xticks(x, [name.replace("_", "\n") for name in methods], fontsize=7)
    plt.ylim(0, 1.0)
    plt.ylabel("Rate")
    plt.title("Combined-stress closed-loop outcomes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "disagreement_task_outcomes.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(x - 0.17, [float(row["protocol_cost"]) for row in combined], width=0.34, label="protocol cost")
    plt.bar(x + 0.17, [float(row["planning_regret_to_oracle"]) for row in combined], width=0.34, label="regret to oracle")
    plt.xticks(x, [name.replace("_", "\n") for name in methods], fontsize=7)
    plt.ylabel("Mean per episode")
    plt.title("Cost and regret expose taxonomy overhead")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "disagreement_cost_regret.png", dpi=180)
    plt.close()

    ablation_names = [row["ablation"] for row in ablations]
    ax = np.arange(len(ablation_names))
    plt.figure(figsize=(11, 5))
    plt.bar(ax - 0.17, [float(row["task_success"]) for row in ablations], width=0.34, label="task success")
    plt.bar(ax + 0.17, [float(row["planning_regret_to_oracle"]) for row in ablations], width=0.34, label="regret")
    plt.xticks(ax, [name.replace("_", "\n") for name in ablation_names], fontsize=7)
    plt.title("Ablations do not isolate a positive type-protocol mechanism")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "disagreement_ablation.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 5))
    for method in [
        "robust_mpc_fallback",
        "failure_aware_rl_recovery",
        "worldbench_diagnostic_probe",
        "ensemble_forecasting_calibrator",
        "proposed_disagreement_protocol",
    ]:
        method_rows = sorted([row for row in sweep if row["method"] == method], key=lambda row: float(row["stress_level"]))
        plt.plot([float(row["stress_level"]) for row in method_rows], [float(row["task_success"]) for row in method_rows], marker="o", label=method.replace("_", " "))
    plt.xlabel("Combined stress level")
    plt.ylabel("Task success")
    plt.title("Stress sweep: simpler recovery/fallback remains competitive")
    plt.ylim(0.0, 1.0)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES / "disagreement_stress_sweep.png", dpi=180)
    plt.close()


def terminal_decision(metrics: list[dict[str, object]], pairwise: list[dict[str, object]], ablations: list[dict[str, object]]) -> dict[str, object]:
    combined = {row["method"]: row for row in metrics if row["split"] == "combined_disagreement_stress"}
    proposed = combined["proposed_disagreement_protocol"]
    non_oracle = {name: row for name, row in combined.items() if name not in {"proposed_disagreement_protocol", "oracle_disagreement_type"}}
    best_success_name, best_success_row = max(non_oracle.items(), key=lambda item: float(item[1]["task_success"]))
    best_unsafe_name, best_unsafe_row = min(non_oracle.items(), key=lambda item: float(item[1]["unsafe_commit_rate"]))
    best_regret_name, _ = min(non_oracle.items(), key=lambda item: float(item[1]["planning_regret_to_oracle"]))
    pair_success = [row for row in pairwise if row["baseline"] == best_success_name and row["metric"] == "task_success"][0]
    pair_unsafe = [row for row in pairwise if row["baseline"] == best_unsafe_name and row["metric"] == "unsafe_commit_rate"][0]
    ablation_by_name = {row["ablation"]: row for row in ablations}
    full_success = float(ablation_by_name["full_disagreement_protocol"]["task_success"])
    full_regret = float(ablation_by_name["full_disagreement_protocol"]["planning_regret_to_oracle"])
    ablation_beats = [
        name
        for name, row in ablation_by_name.items()
        if name != "full_disagreement_protocol"
        and (float(row["task_success"]) >= full_success or float(row["planning_regret_to_oracle"]) <= full_regret)
    ]
    success_gate = pair_success["winner"] == "proposed_disagreement_protocol"
    unsafe_gate = pair_unsafe["winner"] == "proposed_disagreement_protocol"
    recall_gate = float(proposed["useful_disagreement_recall"]) >= max(float(row["useful_disagreement_recall"]) for row in non_oracle.values()) - 0.015
    false_alarm_gate = float(proposed["noise_false_alarm_rate"]) < 0.30
    ablation_gate = not ablation_beats
    decision = "STRONG_REVISE" if all([success_gate, unsafe_gate, recall_gate, false_alarm_gate, ablation_gate]) else "KILL_ARCHIVE"
    reasons = []
    if not success_gate:
        reasons.append(f"proposed does not beat strongest success baseline {best_success_name} ({float(proposed['task_success']):.3f} vs {float(best_success_row['task_success']):.3f})")
    if not unsafe_gate:
        reasons.append(f"proposed does not beat safest baseline {best_unsafe_name} ({float(proposed['unsafe_commit_rate']):.3f} vs {float(best_unsafe_row['unsafe_commit_rate']):.3f})")
    if not false_alarm_gate:
        reasons.append(f"noise false-alarm rate remains high ({float(proposed['noise_false_alarm_rate']):.3f})")
    if not ablation_gate:
        reasons.append("ablations match or beat full on success/regret: " + ", ".join(ablation_beats))
    if not reasons:
        reasons.append("all gates passed")
    return {
        "decision": decision,
        "best_success_baseline": best_success_name,
        "best_unsafe_baseline": best_unsafe_name,
        "best_regret_baseline": best_regret_name,
        "success_gate": success_gate,
        "unsafe_gate": unsafe_gate,
        "recall_gate": recall_gate,
        "false_alarm_gate": false_alarm_gate,
        "ablation_gate": ablation_gate,
        "reasons": reasons,
    }


def write_summary(
    metrics: list[dict[str, object]],
    pairwise: list[dict[str, object]],
    ablations: list[dict[str, object]],
    failure_cases: list[dict[str, object]],
    decision: dict[str, object],
) -> None:
    combined = [row for row in metrics if row["split"] == "combined_disagreement_stress"]
    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("Paper 97: model_disagreement_protocols v4 evidence audit\n")
        handle.write("Terminal decision: " + decision["decision"] + "\n")
        handle.write(
            "Design: 5 robotics tasks x 7 disagreement families x 5 splits x 9 methods, "
            f"{len(SEEDS)} seeds, {EPISODES_PER_GROUP} episodes per seed/task/family/method group.\n"
        )
        handle.write("Claim under test: disagreement helps only when classified into an action protocol (commit, probe, switch controller, abstain, recover).\n\n")
        handle.write("Combined-stress evidence:\n")
        for row in sorted(combined, key=lambda item: float(item["task_success"]), reverse=True):
            handle.write(
                f"- {row['method']}: success={float(row['task_success']):.3f} +/- {float(row['ci95_task_success']):.3f}, "
                f"unsafe={float(row['unsafe_commit_rate']):.3f}, type_acc={float(row['disagreement_family_accuracy']):.3f}, "
                f"recall={float(row['useful_disagreement_recall']):.3f}, noise_false_alarm={float(row['noise_false_alarm_rate']):.3f}, "
                f"cost={float(row['protocol_cost']):.3f}, regret={float(row['planning_regret_to_oracle']):.3f}\n"
            )
        handle.write("\nGate outcomes:\n")
        for key in ["success_gate", "unsafe_gate", "recall_gate", "false_alarm_gate", "ablation_gate"]:
            handle.write(f"- {key}: {decision[key]}\n")
        handle.write("\nTerminal rationale:\n")
        for reason in decision["reasons"]:
            handle.write(f"- {reason}\n")
        handle.write("\nAblation summary:\n")
        for row in sorted(ablations, key=lambda item: float(item["task_success"]), reverse=True):
            handle.write(
                f"- {row['ablation']}: success={float(row['task_success']):.3f}, unsafe={float(row['unsafe_commit_rate']):.3f}, "
                f"regret={float(row['planning_regret_to_oracle']):.3f}, type_acc={float(row['disagreement_family_accuracy']):.3f}\n"
            )
        handle.write("\nRepresentative failure cases:\n")
        for row in failure_cases[:5]:
            handle.write(f"- {row['task']} / {row['family']} seed {row['seed']}: success={row['task_success']}, regret={row['planning_regret_to_oracle']}; {row['failure_mode']}\n")
        handle.write("\nPaired comparison rows: " + str(len(pairwise)) + "\n")
        handle.write("No hardware validation is claimed; this is a local executable surrogate audit.\n")


def assert_no_nan(rows: list[dict[str, object]], name: str) -> None:
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} has non-finite value at {key}: {row}")


def main() -> None:
    raw_rows = [
        simulate_method_group(seed, split, task, family, method, EPISODES_PER_GROUP)
        for split in SPLITS
        for method in METHODS
        for seed in SEEDS
        for task in TASKS
        for family in FAMILIES
    ]
    assert_no_nan(raw_rows, "seed_task_family_metrics")
    metrics = aggregate(raw_rows, ["method", "split"])
    per_task_family = aggregate(raw_rows, ["method", "split", "task", "family"])
    pairwise = pairwise_stats(raw_rows)
    ablation_seed_rows = [
        simulate_ablation_group(seed, task, family, ablation)
        for ablation in ABLATIONS
        for seed in SEEDS
        for task in TASKS
        for family in FAMILIES
    ]
    ablation_metrics = aggregate(ablation_seed_rows, ["ablation"])
    sweep_seed_rows, sweep_summary = stress_sweep()
    failure_cases = build_failure_cases(raw_rows)
    decision = terminal_decision(metrics, pairwise, ablation_metrics)

    write_csv(RESULTS / "seed_task_family_metrics.csv", raw_rows)
    write_csv(RESULTS / "metrics.csv", metrics)
    write_csv(RESULTS / "per_task_family_metrics.csv", per_task_family)
    write_csv(RESULTS / "pairwise_stats.csv", pairwise)
    write_csv(RESULTS / "ablation_seed_metrics.csv", ablation_seed_rows)
    write_csv(RESULTS / "ablation_metrics.csv", ablation_metrics)
    write_csv(RESULTS / "stress_sweep_seed_metrics.csv", sweep_seed_rows)
    write_csv(RESULTS / "stress_sweep.csv", sweep_summary)
    write_csv(RESULTS / "failure_cases.csv", failure_cases)

    combined = [row for row in metrics if row["split"] == "combined_disagreement_stress" and row["method"] != "oracle_disagreement_type"]
    combined.sort(key=lambda row: float(row["task_success"]), reverse=True)
    make_latex_table(
        RESULTS / "combined_stress_table.tex",
        combined,
        [
            ("method", "Method"),
            ("task_success", "Success"),
            ("unsafe_commit_rate", "Unsafe"),
            ("disagreement_family_accuracy", "Type acc."),
            ("useful_disagreement_recall", "Recall"),
            ("noise_false_alarm_rate", "False alarm"),
            ("planning_regret_to_oracle", "Regret"),
        ],
    )
    ablation_metrics.sort(key=lambda row: float(row["task_success"]), reverse=True)
    make_latex_table(
        RESULTS / "ablation_table.tex",
        ablation_metrics,
        [
            ("ablation", "Ablation"),
            ("task_success", "Success"),
            ("unsafe_commit_rate", "Unsafe"),
            ("protocol_cost", "Cost"),
            ("planning_regret_to_oracle", "Regret"),
        ],
    )
    decision_pairs = [
        row
        for row in pairwise
        if row["metric"] in {"task_success", "unsafe_commit_rate", "planning_regret_to_oracle"}
        and row["baseline"] in {decision["best_success_baseline"], decision["best_unsafe_baseline"], decision["best_regret_baseline"]}
    ]
    make_latex_table(
        RESULTS / "pairwise_decision_table.tex",
        decision_pairs,
        [
            ("baseline", "Baseline"),
            ("metric", "Metric"),
            ("proposed_mean", "Proposed"),
            ("baseline_mean", "Baseline"),
            ("mean_diff_proposed_minus_baseline", "Diff"),
            ("ci95_diff", "CI95"),
            ("winner", "Winner"),
        ],
    )
    plot_figures(metrics, ablation_metrics, sweep_summary)
    write_summary(metrics, pairwise, ablation_metrics, failure_cases, decision)
    print(f"Paper 97 evidence audit complete: {decision['decision']}")
    print("Reasons:")
    for reason in decision["reasons"]:
        print("-", reason)
    print("Wrote results to", RESULTS)


if __name__ == "__main__":
    main()
