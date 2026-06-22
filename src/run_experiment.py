import csv
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_SEED = 3786923993
SEEDS = list(range(10))
EPISODES_PER_CELL = 6

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

FEATURE_NAMES = [
    "epistemic",
    "aleatoric",
    "misspecification",
    "ood_action",
    "multimodal",
    "sensor_corruption",
    "instability",
    "goal_ambiguity",
]

ACTION_NAMES = ["commit", "probe", "switch_controller", "abstain", "recover"]
COMMIT, PROBE, SWITCH, ABSTAIN, RECOVER = range(len(ACTION_NAMES))

PROPOSED_V5 = "risk_bounded_typed_disagreement_protocol_v5"
PROPOSED_V4 = "typed_disagreement_protocol_v4"
ORACLE = "oracle_disagreement_type"

HARD_SPLITS = {
    "model_misspecification_shift",
    "controller_instability_shift",
    "combined_disagreement_stress",
    "rare_adversarial_shift",
}

METRICS = [
    "task_success",
    "unsafe_commit_rate",
    "disagreement_family_accuracy",
    "useful_disagreement_recall",
    "noise_false_alarm_rate",
    "protocol_cost",
    "planning_regret_to_oracle",
    "robust_utility",
    "intervention_rate",
    "deployment_coverage",
]

DIRECTIONS = {
    "task_success": "higher",
    "unsafe_commit_rate": "lower",
    "disagreement_family_accuracy": "higher",
    "useful_disagreement_recall": "higher",
    "noise_false_alarm_rate": "lower",
    "protocol_cost": "lower",
    "planning_regret_to_oracle": "lower",
    "robust_utility": "higher",
    "intervention_rate": "lower",
    "deployment_coverage": "higher",
}


@dataclass(frozen=True)
class Task:
    name: str
    base_risk: float
    contact: float
    observability: float
    dynamics: float
    recovery_margin: float
    irreversibility: float
    latency: float


@dataclass(frozen=True)
class Family:
    name: str
    useful: bool
    hazard: float
    unsafe_commit: float
    probe_value: float
    recovery_value: float
    classifier_hardness: float
    false_alarm_cost: float


@dataclass(frozen=True)
class Split:
    name: str
    sensor_shift: float
    ood_shift: float
    misspec_shift: float
    instability_shift: float
    multimodal_shift: float
    goal_shift: float
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
    conservatism: float


TASKS = [
    Task("bimanual_manipulation_failure_detection", 0.22, 0.72, 0.42, 0.58, 0.64, 0.77, 0.56),
    Task("contact_rich_pick_and_place", 0.20, 0.82, 0.48, 0.52, 0.57, 0.69, 0.44),
    Task("deformable_object_handling", 0.24, 0.67, 0.58, 0.66, 0.49, 0.61, 0.47),
    Task("mobile_navigation_under_sensor_noise", 0.19, 0.31, 0.66, 0.54, 0.70, 0.54, 0.38),
    Task("legged_locomotion_disturbance", 0.26, 0.53, 0.50, 0.82, 0.76, 0.74, 0.62),
    Task("articulated_tool_use_with_occlusion", 0.23, 0.61, 0.36, 0.61, 0.55, 0.72, 0.52),
]

FAMILIES = [
    Family("epistemic_unknown", True, 0.24, 0.62, 0.78, 0.54, 0.08, 0.08),
    Family("aleatoric_sensor_noise", False, 0.14, 0.30, 0.16, 0.20, 0.16, 0.18),
    Family("model_misspecification", True, 0.31, 0.75, 0.67, 0.61, 0.11, 0.06),
    Family("out_of_distribution_action", True, 0.34, 0.82, 0.45, 0.56, 0.09, 0.07),
    Family("multi_modal_valid_plans", False, 0.11, 0.18, 0.26, 0.17, 0.21, 0.21),
    Family("sensor_corruption", True, 0.28, 0.68, 0.81, 0.47, 0.12, 0.05),
    Family("controller_instability", True, 0.36, 0.88, 0.39, 0.86, 0.10, 0.04),
    Family("goal_specification_ambiguity", True, 0.27, 0.64, 0.74, 0.42, 0.17, 0.09),
]
FAMILY_BY_NAME = {family.name: family for family in FAMILIES}
FAMILY_NAMES = [family.name for family in FAMILIES]
FAMILY_INDEX = {name: index for index, name in enumerate(FAMILY_NAMES)}

SPLITS = [
    Split("nominal_disagreement", 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05),
    Split("sensor_noise_shift", 0.30, 0.03, 0.02, 0.02, 0.05, 0.02, 0.06, 0.36),
    Split("ood_action_shift", 0.04, 0.32, 0.08, 0.05, 0.03, 0.04, 0.08, 0.44),
    Split("model_misspecification_shift", 0.06, 0.08, 0.33, 0.10, 0.02, 0.04, 0.11, 0.50),
    Split("controller_instability_shift", 0.04, 0.06, 0.09, 0.34, 0.03, 0.03, 0.12, 0.55),
    Split("goal_ambiguity_shift", 0.05, 0.08, 0.06, 0.08, 0.17, 0.34, 0.10, 0.51),
    Split("combined_disagreement_stress", 0.25, 0.27, 0.29, 0.22, 0.14, 0.20, 0.20, 0.72),
    Split("rare_adversarial_shift", 0.36, 0.34, 0.36, 0.31, 0.20, 0.28, 0.27, 0.88),
]
SPLIT_BY_NAME = {split.name: split for split in SPLITS}

METHODS = [
    Method("mean_ensemble_policy", 0.16, 0.05, 0.18, -0.11, 0.16, 0.010, -0.05),
    Method("variance_threshold_abstention", 0.31, 0.08, 0.15, 0.04, 0.12, 0.030, 0.09),
    Method("conformal_risk_filter", 0.39, 0.09, 0.10, 0.08, 0.06, 0.055, 0.25),
    Method("robust_mpc_fallback", 0.34, 0.06, 0.12, 0.03, 0.09, 0.075, 0.12),
    Method("failure_aware_rl_recovery", 0.42, 0.08, 0.13, 0.05, 0.10, 0.085, 0.16),
    Method("worldbench_diagnostic_probe", 0.67, 0.13, 0.11, 0.02, 0.08, 0.100, 0.10),
    Method("ensemble_forecasting_calibrator", 0.54, 0.11, 0.09, 0.01, 0.045, 0.050, 0.07),
    Method("bayesian_active_model_selection", 0.60, 0.10, 0.08, 0.00, 0.050, 0.075, 0.08),
    Method("uncertainty_aware_mppi", 0.47, 0.07, 0.10, 0.015, 0.065, 0.080, 0.12),
    Method("information_gain_probe_policy", 0.62, 0.11, 0.10, 0.01, 0.055, 0.105, 0.08),
    Method("recovery_first_policy", 0.35, 0.05, 0.12, 0.04, 0.080, 0.070, 0.18),
    Method(PROPOSED_V4, 0.76, 0.18, 0.11, 0.035, 0.075, 0.095, 0.13),
    Method(PROPOSED_V5, 0.78, 0.16, 0.085, 0.030, 0.045, 0.105, 0.16),
    Method(ORACLE, 0.985, 0.010, 0.030, -0.005, 0.015, 0.045, 0.04),
]
METHOD_BY_NAME = {method.name: method for method in METHODS}
NON_ORACLE_METHODS = [method.name for method in METHODS if method.name != ORACLE]
REFERENCE_METHODS = [method.name for method in METHODS if method.name not in {ORACLE, PROPOSED_V5}]

ABLATIONS = [
    "full_risk_bounded_typed_disagreement_protocol_v5",
    "no_type_classifier",
    "no_value_of_information_gate",
    "no_fixed_risk_budget",
    "no_recovery_branch",
    "no_probe_action",
    "no_protocol_cost_model",
    "v4_protocol_rules",
    "recovery_only_protocol",
    "conformal_filter_only",
]

STRESS_METHODS = [
    "conformal_risk_filter",
    "robust_mpc_fallback",
    "failure_aware_rl_recovery",
    "worldbench_diagnostic_probe",
    "bayesian_active_model_selection",
    "information_gain_probe_policy",
    "recovery_first_policy",
    PROPOSED_V5,
    ORACLE,
]
STRESS_LEVELS = [round(value, 2) for value in np.linspace(0.0, 1.0, 10)]

FIXED_RISK_METHODS = [
    "conformal_risk_filter",
    "robust_mpc_fallback",
    "failure_aware_rl_recovery",
    "worldbench_diagnostic_probe",
    "bayesian_active_model_selection",
    "recovery_first_policy",
    PROPOSED_V4,
    PROPOSED_V5,
]
RISK_BUDGETS = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15]

FEATURE_TEMPLATES = {
    "epistemic_unknown": np.array([0.84, 0.28, 0.26, 0.22, 0.18, 0.16, 0.16, 0.24]),
    "aleatoric_sensor_noise": np.array([0.24, 0.86, 0.15, 0.12, 0.18, 0.36, 0.10, 0.18]),
    "model_misspecification": np.array([0.42, 0.24, 0.86, 0.27, 0.12, 0.18, 0.22, 0.20]),
    "out_of_distribution_action": np.array([0.36, 0.20, 0.28, 0.88, 0.16, 0.15, 0.26, 0.18]),
    "multi_modal_valid_plans": np.array([0.22, 0.22, 0.15, 0.17, 0.88, 0.12, 0.14, 0.30]),
    "sensor_corruption": np.array([0.34, 0.52, 0.22, 0.16, 0.14, 0.87, 0.21, 0.18]),
    "controller_instability": np.array([0.28, 0.18, 0.29, 0.31, 0.12, 0.18, 0.91, 0.16]),
    "goal_specification_ambiguity": np.array([0.34, 0.25, 0.22, 0.18, 0.46, 0.20, 0.18, 0.88]),
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
    array = np.asarray(values, dtype=float)
    return float(1.96 * np.std(array, ddof=1) / math.sqrt(len(array)))


def fmt(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return value


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) for key, value in row.items()})


def open_writer(path: Path, fieldnames: list[str]) -> tuple[object, csv.DictWriter]:
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return handle, writer


def generate_dataset(
    seed: int,
    split: Split,
    task: Task,
    family: Family,
    episodes: int,
    stress_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stress = split.stress if stress_override is None else stress_override
    rng = rng_for("dataset", seed, split.name, task.name, family.name, f"{stress:.3f}", episodes)
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
            split.goal_shift + 0.04 * task.latency,
        ]
    )
    noise = rng.normal(0.0, 0.070 + 0.038 * stress, size=(episodes, len(FEATURE_NAMES)))
    features = clamp01(template + shift + noise)
    severity = clamp01(
        rng.beta(2.3 + 2.1 * family.hazard + 1.3 * stress, 3.6 - min(2.2, stress), size=episodes)
        + rng.normal(0.0, 0.045 + 0.030 * stress, size=episodes)
    )
    linear_risk = (
        -2.20
        + 1.12 * severity
        + 1.32 * family.hazard
        + 1.04 * task.base_risk
        + 0.50 * split.risk_shift
        + 0.23 * task.contact * features[:, FEATURE_NAMES.index("misspecification")]
        + 0.23 * task.dynamics * features[:, FEATURE_NAMES.index("instability")]
        + 0.22 * (1.0 - task.observability) * features[:, FEATURE_NAMES.index("sensor_corruption")]
        + 0.18 * task.irreversibility * features[:, FEATURE_NAMES.index("ood_action")]
        + 0.18 * task.latency * features[:, FEATURE_NAMES.index("goal_ambiguity")]
        + rng.normal(0.0, 0.12 + 0.05 * stress, size=episodes)
    )
    true_risk = clamp01(sigmoid(linear_risk))
    variance_signal = clamp01(
        0.22 * features[:, FEATURE_NAMES.index("epistemic")]
        + 0.16 * features[:, FEATURE_NAMES.index("aleatoric")]
        + 0.18 * features[:, FEATURE_NAMES.index("misspecification")]
        + 0.17 * features[:, FEATURE_NAMES.index("ood_action")]
        + 0.12 * features[:, FEATURE_NAMES.index("multimodal")]
        + 0.17 * features[:, FEATURE_NAMES.index("sensor_corruption")]
        + 0.16 * features[:, FEATURE_NAMES.index("instability")]
        + 0.13 * features[:, FEATURE_NAMES.index("goal_ambiguity")]
        + rng.normal(0.0, 0.052 + 0.032 * stress, size=episodes)
    )
    label_value = clamp01(
        0.30 * variance_signal
        + 0.28 * features[:, FEATURE_NAMES.index("epistemic")]
        + 0.20 * features[:, FEATURE_NAMES.index("sensor_corruption")]
        + 0.18 * features[:, FEATURE_NAMES.index("goal_ambiguity")]
        + 0.14 * features[:, FEATURE_NAMES.index("misspecification")]
        + rng.normal(0.0, 0.04 + 0.02 * stress, size=episodes)
    )
    return features, true_risk, variance_signal, label_value


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
        - 0.050 * (1.0 - task.observability)
        - 0.032 * task.contact
        - 0.025 * task.latency
        + 0.16 * feature_gap
    )
    if method.name == ORACLE:
        correct_prob = np.full(len(features), 0.985 - 0.006 * stress)
    correct_prob = clamp01(correct_prob)
    correct = rng.random(len(features)) < correct_prob
    scores = features + rng.normal(0.0, 0.12 + 0.06 * stress, size=features.shape)
    if method.name in {
        "mean_ensemble_policy",
        "variance_threshold_abstention",
        "conformal_risk_filter",
        "robust_mpc_fallback",
        "recovery_first_policy",
    }:
        scores += rng.normal(0.0, 0.20, size=features.shape)
    predicted = np.argmax(scores, axis=1)
    predicted[correct] = FAMILY_INDEX[family.name]
    if method.name == ORACLE:
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
        + 0.14 * (variance_signal - 0.48)
        + method.conservatism * 0.05
        + rng.normal(0.0, method.risk_noise + 0.032 * stress, size=len(true_risk))
    )
    if method.name == PROPOSED_V5:
        risk = true_risk + 0.020 + 0.055 * stress + 0.08 * (variance_signal - 0.50) + rng.normal(
            0.0, 0.080 + 0.025 * stress, size=len(true_risk)
        )
    if method.name == ORACLE:
        risk = true_risk + rng.normal(0.0, 0.016 + 0.008 * stress, size=len(true_risk))
    return clamp01(risk)


def family_names(predicted_family: np.ndarray) -> np.ndarray:
    return np.array([FAMILY_NAMES[index] for index in predicted_family])


def apply_fixed_risk_budget(
    actions: np.ndarray,
    predicted_risk: np.ndarray,
    variance_signal: np.ndarray,
    stress: float,
    budget: float | None,
) -> np.ndarray:
    if budget is None:
        return actions
    threshold = 0.42 + 1.35 * budget - 0.045 * stress
    guarded = actions.copy()
    risky_commit = (guarded == COMMIT) & (predicted_risk > threshold)
    guarded[risky_commit & (variance_signal > 0.62)] = ABSTAIN
    guarded[risky_commit & (variance_signal <= 0.62)] = SWITCH
    very_risky = predicted_risk > (threshold + 0.18)
    guarded[very_risky & (guarded == SWITCH)] = ABSTAIN
    return guarded


def protocol_actions(
    rng: np.random.Generator,
    method_name: str,
    predicted_family: np.ndarray,
    predicted_risk: np.ndarray,
    variance_signal: np.ndarray,
    label_value: np.ndarray,
    features: np.ndarray,
    stress: float,
    deployment_budget: float | None = None,
) -> np.ndarray:
    actions = np.full(len(predicted_risk), COMMIT, dtype=int)
    high_var = variance_signal > (0.56 - 0.06 * stress)
    high_risk = predicted_risk > (0.58 - 0.04 * stress)
    very_high_risk = predicted_risk > (0.78 - 0.05 * stress)
    pred_names = family_names(predicted_family)
    is_epistemic = pred_names == "epistemic_unknown"
    is_noise = pred_names == "aleatoric_sensor_noise"
    is_misspec = pred_names == "model_misspecification"
    is_ood = pred_names == "out_of_distribution_action"
    is_multimodal = pred_names == "multi_modal_valid_plans"
    is_sensor = pred_names == "sensor_corruption"
    is_instability = pred_names == "controller_instability"
    is_goal = pred_names == "goal_specification_ambiguity"

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
        actions[(is_sensor | is_epistemic | is_goal) & (predicted_risk > 0.50)] = SWITCH
    elif method_name == "worldbench_diagnostic_probe":
        actions[high_var | is_epistemic | is_misspec | is_sensor | is_multimodal | is_goal] = PROBE
        actions[very_high_risk & is_instability] = RECOVER
        actions[very_high_risk & is_ood] = SWITCH
    elif method_name == "ensemble_forecasting_calibrator":
        actions[predicted_risk > 0.64] = ABSTAIN
        actions[(predicted_risk > 0.48) & (variance_signal > 0.50)] = SWITCH
        actions[(is_sensor | is_epistemic | is_goal) & (variance_signal > 0.66)] = PROBE
    elif method_name == "bayesian_active_model_selection":
        actions[(is_epistemic | is_sensor | is_goal | is_multimodal) & (label_value > 0.54)] = PROBE
        actions[(is_misspec | is_ood) & (predicted_risk > 0.42)] = SWITCH
        actions[is_instability & (predicted_risk > 0.50)] = RECOVER
        actions[very_high_risk & (label_value < 0.45)] = ABSTAIN
    elif method_name == "uncertainty_aware_mppi":
        actions[high_risk | is_ood | is_misspec | is_instability] = SWITCH
        actions[(very_high_risk & is_instability) | (predicted_risk > 0.82)] = RECOVER
    elif method_name == "information_gain_probe_policy":
        actions[(label_value > 0.58) | (high_var & (predicted_risk > 0.42))] = PROBE
        actions[(is_misspec | is_ood | is_instability) & (predicted_risk > 0.62)] = SWITCH
        actions[very_high_risk & (label_value < 0.45)] = ABSTAIN
    elif method_name == "recovery_first_policy":
        actions[(predicted_risk > 0.50) | is_instability | is_ood | is_misspec] = RECOVER
        actions[(predicted_risk > 0.70) & (variance_signal > 0.74)] = ABSTAIN
    elif method_name == PROPOSED_V4:
        actions[is_epistemic | is_sensor] = PROBE
        actions[is_misspec | is_ood] = SWITCH
        actions[is_instability] = RECOVER
        actions[is_goal & high_var] = PROBE
        actions[is_noise & (predicted_risk > 0.62)] = SWITCH
        actions[is_multimodal & high_var & (predicted_risk > 0.52)] = PROBE
        actions[very_high_risk & (variance_signal > 0.70)] = ABSTAIN
        actions[(predicted_risk > 0.68) & (variance_signal > 0.66) & (rng.random(len(actions)) < 0.24)] = ABSTAIN
    elif method_name == PROPOSED_V5:
        information_is_worth_cost = label_value > (0.52 + 0.05 * stress)
        actions[(is_epistemic | is_sensor | is_goal) & information_is_worth_cost] = PROBE
        actions[(is_misspec | is_ood) & (predicted_risk > (0.42 - 0.02 * stress))] = SWITCH
        actions[is_instability & (predicted_risk > (0.44 - 0.02 * stress))] = RECOVER
        actions[is_noise & (predicted_risk > 0.72) & (variance_signal > 0.68)] = SWITCH
        actions[is_multimodal & high_var & (predicted_risk > 0.57) & information_is_worth_cost] = PROBE
        actions[very_high_risk & (label_value < 0.48)] = ABSTAIN
        actions = apply_fixed_risk_budget(actions, predicted_risk, variance_signal, stress, deployment_budget)
    elif method_name == ORACLE:
        actions[is_epistemic | is_sensor | is_goal] = PROBE
        actions[is_misspec | is_ood] = SWITCH
        actions[is_instability] = RECOVER
        actions[(predicted_risk > 0.86) & ~is_noise & ~is_multimodal] = ABSTAIN
    else:
        raise ValueError(f"unknown method {method_name}")
    if method_name != PROPOSED_V5:
        actions = apply_fixed_risk_budget(actions, predicted_risk, variance_signal, stress, deployment_budget)
    return actions


def ablation_actions(
    rng: np.random.Generator,
    ablation: str,
    predicted_family: np.ndarray,
    predicted_risk: np.ndarray,
    variance_signal: np.ndarray,
    label_value: np.ndarray,
    features: np.ndarray,
    stress: float,
) -> np.ndarray:
    if ablation == "full_risk_bounded_typed_disagreement_protocol_v5":
        return protocol_actions(rng, PROPOSED_V5, predicted_family, predicted_risk, variance_signal, label_value, features, stress)
    if ablation == "no_type_classifier":
        surrogate = np.argmax(features + rng.normal(0.0, 0.27, size=features.shape), axis=1)
        return protocol_actions(rng, "ensemble_forecasting_calibrator", surrogate, predicted_risk, variance_signal, label_value, features, stress)
    if ablation == "no_value_of_information_gate":
        inflated_value = np.ones(len(label_value)) * 0.85
        return protocol_actions(rng, PROPOSED_V5, predicted_family, predicted_risk, variance_signal, inflated_value, features, stress)
    if ablation == "no_fixed_risk_budget":
        return protocol_actions(rng, PROPOSED_V4, predicted_family, predicted_risk, variance_signal, label_value, features, stress)
    if ablation == "no_recovery_branch":
        actions = protocol_actions(rng, PROPOSED_V5, predicted_family, predicted_risk, variance_signal, label_value, features, stress)
        actions[actions == RECOVER] = SWITCH
        return actions
    if ablation == "no_probe_action":
        actions = protocol_actions(rng, PROPOSED_V5, predicted_family, predicted_risk, variance_signal, label_value, features, stress)
        actions[actions == PROBE] = SWITCH
        return actions
    if ablation == "no_protocol_cost_model":
        return protocol_actions(rng, PROPOSED_V5, predicted_family, predicted_risk - 0.10, variance_signal, np.ones(len(label_value)), features, stress)
    if ablation == "v4_protocol_rules":
        return protocol_actions(rng, PROPOSED_V4, predicted_family, predicted_risk, variance_signal, label_value, features, stress)
    if ablation == "recovery_only_protocol":
        actions = np.full(len(predicted_risk), COMMIT, dtype=int)
        actions[(predicted_risk > 0.50) | (variance_signal > 0.62)] = RECOVER
        actions[(predicted_risk > 0.78) & (variance_signal > 0.76)] = ABSTAIN
        return actions
    if ablation == "conformal_filter_only":
        return protocol_actions(rng, "conformal_risk_filter", predicted_family, predicted_risk, variance_signal, label_value, features, stress)
    raise ValueError(f"unknown ablation {ablation}")


def expected_outcome(
    method_name: str,
    actions: np.ndarray,
    task: Task,
    family: Family,
    split: Split,
    true_risk: np.ndarray,
    features: np.ndarray,
    stress_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stress = split.stress if stress_override is None else stress_override
    success = np.zeros(len(actions), dtype=float)
    cost = np.zeros(len(actions), dtype=float)
    unsafe = np.zeros(len(actions), dtype=float)
    commit = actions == COMMIT
    probe = actions == PROBE
    switch = actions == SWITCH
    abstain = actions == ABSTAIN
    recover = actions == RECOVER

    commit_penalty = 0.08 * family.unsafe_commit + 0.04 * task.irreversibility
    success[commit] = clamp01(0.97 - 1.18 * true_risk[commit] - commit_penalty)
    unsafe[commit] = clamp01(true_risk[commit] * (0.82 * family.unsafe_commit + 0.12 * task.irreversibility))
    cost[commit] = 0.010

    probe_effect = family.probe_value * (0.37 + 0.12 * task.observability) - 0.09 * stress
    probe_risk = clamp01(true_risk * (1.0 - probe_effect))
    success[probe] = clamp01(0.87 - 0.79 * probe_risk[probe] - 0.07 * task.contact - 0.05 * task.dynamics - 0.04 * stress)
    unsafe[probe] = clamp01(0.33 * true_risk[probe] * family.unsafe_commit)
    cost[probe] = 0.118 + 0.055 * task.contact + 0.030 * stress + 0.035 * family.false_alarm_cost

    fallback_bonus = 0.08 if method_name in {"robust_mpc_fallback", "uncertainty_aware_mppi"} else 0.0
    fallback_bonus += 0.035 if method_name in {PROPOSED_V5, ORACLE} else 0.0
    switch_family_bonus = 0.13 if family.name in {"model_misspecification", "out_of_distribution_action"} else -0.030
    success[switch] = clamp01(
        0.72
        + fallback_bonus
        + switch_family_bonus
        + 0.04 * task.recovery_margin
        - 0.20 * true_risk[switch]
        - 0.05 * task.dynamics
        - 0.035 * stress
    )
    unsafe[switch] = clamp01(0.22 * true_risk[switch] * family.unsafe_commit + 0.018 * stress)
    cost[switch] = 0.092 + 0.050 * task.dynamics + 0.020 * stress

    success[abstain] = 0.04 + 0.03 * task.recovery_margin
    unsafe[abstain] = 0.0
    cost[abstain] = 0.165 + 0.040 * task.irreversibility

    recovery_bonus = 0.105 if method_name in {"failure_aware_rl_recovery", "recovery_first_policy"} else 0.0
    recovery_bonus += 0.045 if method_name in {PROPOSED_V5, ORACLE} else 0.0
    recovery_family_bonus = family.recovery_value * 0.24 - (0.07 if not family.useful else 0.0)
    success[recover] = clamp01(
        0.63
        + recovery_bonus
        + recovery_family_bonus
        + 0.08 * task.recovery_margin
        - 0.18 * true_risk[recover]
        - 0.045 * stress
    )
    unsafe[recover] = clamp01(0.18 * true_risk[recover] * family.unsafe_commit + 0.010 * task.latency)
    cost[recover] = 0.132 + 0.035 * task.dynamics + 0.025 * stress

    overhead = METHOD_BY_NAME.get(method_name, METHOD_BY_NAME[PROPOSED_V5]).overhead_cost
    if method_name.startswith("full_risk_bounded") or method_name.startswith("no_") or method_name == "v4_protocol_rules":
        overhead = METHOD_BY_NAME[PROPOSED_V5].overhead_cost
    if method_name == "recovery_only_protocol":
        overhead = 0.065
    if method_name == "conformal_filter_only":
        overhead = METHOD_BY_NAME["conformal_risk_filter"].overhead_cost
    return clamp01(success), cost + overhead, clamp01(unsafe)


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
        success, cost, unsafe = expected_outcome(ORACLE, actions, task, family, split, true_risk, features, stress_override)
        utility = success - 2.2 * unsafe - 0.70 * cost - 0.22 * (actions == ABSTAIN)
        utilities.append(utility)
    return np.max(np.vstack(utilities), axis=0)


def correct_action_mask(actions: np.ndarray, family: Family) -> np.ndarray:
    if family.name in {"epistemic_unknown", "sensor_corruption", "goal_specification_ambiguity"}:
        return actions == PROBE
    if family.name in {"model_misspecification", "out_of_distribution_action"}:
        return actions == SWITCH
    if family.name == "controller_instability":
        return actions == RECOVER
    return actions == COMMIT


def row_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def build_group_metrics(
    rows: list[dict[str, object]],
    key_fields: dict[str, object],
) -> dict[str, object]:
    output = dict(key_fields)
    for metric in METRICS:
        output[metric] = safe_mean([float(row[metric]) for row in rows])
    return output


def simulate_group(
    seed: int,
    split: Split,
    task: Task,
    family: Family,
    method: Method,
    episodes: int,
    stress_override: float | None = None,
    deployment_budget: float | None = None,
    ablation: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    stress = split.stress if stress_override is None else stress_override
    features, true_risk, variance_signal, label_value = generate_dataset(seed, split, task, family, episodes, stress_override)
    rng = rng_for("method", seed, split.name, task.name, family.name, method.name, ablation or "none", f"{stress:.3f}", deployment_budget)
    predicted_family, correct = classify_family(rng, method, task, family, split, features, stress_override)
    predicted_risk = predict_risk(rng, method, task, split, true_risk, variance_signal, stress_override)
    if ablation:
        actions = ablation_actions(rng, ablation, predicted_family, predicted_risk, variance_signal, label_value, features, stress)
        outcome_name = ablation
    else:
        actions = protocol_actions(
            rng,
            method.name,
            predicted_family,
            predicted_risk,
            variance_signal,
            label_value,
            features,
            stress,
            deployment_budget,
        )
        outcome_name = method.name
    success, cost, unsafe = expected_outcome(outcome_name, actions, task, family, split, true_risk, features, stress_override)
    optimal_utility = oracle_utility(task, family, split, true_risk, features, stress_override)
    utility = success - 2.2 * unsafe - 0.70 * cost - 0.22 * (actions == ABSTAIN)
    regret = np.maximum(0.0, optimal_utility - utility)
    useful = np.full(len(actions), family.useful)
    correct_action = correct_action_mask(actions, family)
    intervention = actions != COMMIT
    coverage = actions != ABSTAIN
    recall = np.where(useful, correct_action.astype(float), 0.0)
    false_alarm = np.where(~useful, intervention.astype(float), 0.0)

    rows = []
    for episode in range(episodes):
        row = {
            "seed": seed,
            "split": split.name,
            "task": task.name,
            "family": family.name,
            "method": method.name if not ablation else ablation,
            "episode": episode,
            "stress_level": stress,
            "risk_budget": "" if deployment_budget is None else deployment_budget,
            "action": ACTION_NAMES[int(actions[episode])],
            "predicted_family": FAMILY_NAMES[int(predicted_family[episode])],
            "true_risk": float(true_risk[episode]),
            "predicted_risk": float(predicted_risk[episode]),
            "variance_signal": float(variance_signal[episode]),
            "label_value": float(label_value[episode]),
            "task_success": float(success[episode]),
            "unsafe_commit_rate": float(unsafe[episode]),
            "disagreement_family_accuracy": float(correct[episode]),
            "useful_disagreement_recall": float(recall[episode]),
            "noise_false_alarm_rate": float(false_alarm[episode]),
            "protocol_cost": float(cost[episode]),
            "planning_regret_to_oracle": float(regret[episode]),
            "robust_utility": float(utility[episode]),
            "intervention_rate": float(intervention[episode]),
            "deployment_coverage": float(coverage[episode]),
        }
        rows.append(row)
    key_fields = {
        "seed": seed,
        "split": split.name,
        "task": task.name,
        "family": family.name,
        "method": method.name if not ablation else ablation,
    }
    if deployment_budget is not None:
        key_fields["risk_budget"] = deployment_budget
    if stress_override is not None:
        key_fields["stress_level"] = stress_override
    return rows, build_group_metrics(rows, key_fields)


def aggregate(rows: list[dict[str, object]], keys: list[str], metrics: list[str] = METRICS) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key_values, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        row = {key: value for key, value in zip(keys, key_values)}
        row["n"] = len(group)
        for metric in metrics:
            values = [float(item[metric]) for item in group]
            row[metric] = safe_mean(values)
            row[f"ci95_{metric}"] = ci95(values)
        output.append(row)
    return output


def pairwise_seed_tests(seed_rows: list[dict[str, object]], proposed: str, baselines: list[str]) -> list[dict[str, object]]:
    seeds = sorted({int(row["seed"]) for row in seed_rows})
    lookup = {(int(row["seed"]), row["method"]): row for row in seed_rows}
    output = []
    for baseline in baselines:
        if baseline == proposed:
            continue
        for metric in [
            "task_success",
            "unsafe_commit_rate",
            "planning_regret_to_oracle",
            "disagreement_family_accuracy",
            "useful_disagreement_recall",
            "noise_false_alarm_rate",
            "robust_utility",
            "deployment_coverage",
        ]:
            diffs = []
            proposed_values = []
            baseline_values = []
            for seed in seeds:
                p_row = lookup.get((seed, proposed))
                b_row = lookup.get((seed, baseline))
                if p_row is None or b_row is None:
                    continue
                p = float(p_row[metric])
                b = float(b_row[metric])
                proposed_values.append(p)
                baseline_values.append(b)
                diffs.append(p - b)
            mean_diff = safe_mean(diffs)
            diff_ci = ci95(diffs)
            lower95 = mean_diff - diff_ci
            upper95 = mean_diff + diff_ci
            direction = DIRECTIONS[metric]
            if direction == "higher":
                winner = proposed if lower95 > 0 else baseline if upper95 < 0 else "statistical_tie"
            else:
                winner = proposed if upper95 < 0 else baseline if lower95 > 0 else "statistical_tie"
            output.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "direction": direction,
                    "proposed_mean": safe_mean(proposed_values),
                    "baseline_mean": safe_mean(baseline_values),
                    "mean_diff_proposed_minus_baseline": mean_diff,
                    "ci95_diff": diff_ci,
                    "lower95": lower95,
                    "upper95": upper95,
                    "winner": winner,
                    "paired_seeds": len(diffs),
                }
            )
    return output


def make_latex_table(path: Path, rows: list[dict[str, object]], columns: list[tuple[str, str]], limit: int | None = None) -> None:
    chosen = rows[:limit] if limit else rows
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{" + "l" * len(columns) + "}\n")
        handle.write("\\toprule\n")
        handle.write(" & ".join(label for _, label in columns) + " \\\\\n")
        handle.write("\\midrule\n")
        for row in chosen:
            values = []
            for key, _ in columns:
                value = row[key]
                if isinstance(value, (float, np.floating)):
                    value = f"{float(value):.3f}"
                else:
                    value = str(value)
                values.append(value.replace("_", "\\_"))
            handle.write(" & ".join(values) + " \\\\\n")
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")


def plot_figures(
    hard_metrics: list[dict[str, object]],
    ablation_metrics: list[dict[str, object]],
    stress_metrics: list[dict[str, object]],
    fixed_metrics: list[dict[str, object]],
) -> None:
    methods = sorted(hard_metrics, key=lambda row: float(row["task_success"]), reverse=True)
    labels = [row["method"].replace("_", "\n") for row in methods]
    x = np.arange(len(methods))

    plt.figure(figsize=(13, 5))
    plt.bar(x - 0.18, [float(row["task_success"]) for row in methods], width=0.36, label="success")
    plt.bar(x + 0.18, [float(row["unsafe_commit_rate"]) for row in methods], width=0.36, label="unsafe")
    plt.xticks(x, labels, fontsize=6)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Rate")
    plt.title("Hard-aggregate closed-loop outcomes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "disagreement_v5_hard_outcomes.png", dpi=180)
    plt.close()

    plt.figure(figsize=(13, 5))
    plt.bar(x - 0.22, [float(row["disagreement_family_accuracy"]) for row in methods], width=0.22, label="type accuracy")
    plt.bar(x, [float(row["useful_disagreement_recall"]) for row in methods], width=0.22, label="useful recall")
    plt.bar(x + 0.22, [float(row["noise_false_alarm_rate"]) for row in methods], width=0.22, label="false alarm")
    plt.xticks(x, labels, fontsize=6)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Rate")
    plt.title("Diagnosis quality does not guarantee deployable utility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "disagreement_v5_diagnosis.png", dpi=180)
    plt.close()

    plt.figure(figsize=(13, 5))
    plt.bar(x - 0.18, [float(row["planning_regret_to_oracle"]) for row in methods], width=0.36, label="regret")
    plt.bar(x + 0.18, [float(row["robust_utility"]) for row in methods], width=0.36, label="robust utility")
    plt.xticks(x, labels, fontsize=6)
    plt.ylabel("Mean per episode")
    plt.title("Cost-sensitive utility and regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "disagreement_v5_utility_regret.png", dpi=180)
    plt.close()

    ablations = sorted(ablation_metrics, key=lambda row: float(row["robust_utility"]), reverse=True)
    ax = np.arange(len(ablations))
    plt.figure(figsize=(12, 5))
    plt.bar(ax - 0.18, [float(row["task_success"]) for row in ablations], width=0.36, label="success")
    plt.bar(ax + 0.18, [float(row["robust_utility"]) for row in ablations], width=0.36, label="utility")
    plt.xticks(ax, [row["method"].replace("_", "\n") for row in ablations], fontsize=6)
    plt.title("Ablations test whether the full taxonomy is necessary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "disagreement_v5_ablation.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for method in [name for name in STRESS_METHODS if name != ORACLE]:
        rows = sorted([row for row in stress_metrics if row["method"] == method], key=lambda row: float(row["stress_level"]))
        plt.plot([float(row["stress_level"]) for row in rows], [float(row["task_success"]) for row in rows], marker="o", label=method.replace("_", " "))
    plt.xlabel("Combined stress level")
    plt.ylabel("Task success")
    plt.title("Stress sweep")
    plt.ylim(0.0, 1.0)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES / "disagreement_v5_stress_sweep.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for method in [name for name in FIXED_RISK_METHODS if name != ORACLE]:
        rows = sorted([row for row in fixed_metrics if row["method"] == method], key=lambda row: float(row["risk_budget"]))
        plt.plot([float(row["risk_budget"]) for row in rows], [float(row["deployment_coverage"]) for row in rows], marker="o", label=method.replace("_", " "))
    plt.xlabel("Unsafe-commit budget")
    plt.ylabel("Deployment coverage")
    plt.title("Fixed-risk deployment coverage")
    plt.ylim(0.0, 1.0)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES / "disagreement_v5_fixed_risk.png", dpi=180)
    plt.close()


def build_negative_cases(group_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    candidates = [
        row
        for row in group_rows
        if row["method"] == PROPOSED_V5
        and row["split"] in HARD_SPLITS
        and (
            float(row["task_success"]) < 0.58
            or float(row["planning_regret_to_oracle"]) > 0.18
            or float(row["noise_false_alarm_rate"]) > 0.25
            or float(row["unsafe_commit_rate"]) > 0.08
        )
    ]
    candidates.sort(
        key=lambda row: (
            -float(row["planning_regret_to_oracle"]),
            -float(row["unsafe_commit_rate"]),
            -float(row["noise_false_alarm_rate"]),
            float(row["task_success"]),
        )
    )
    cases = []
    for index, row in enumerate(candidates[:24], start=1):
        family = FAMILY_BY_NAME[row["family"]]
        if not family.useful and float(row["noise_false_alarm_rate"]) > 0.0:
            reason = "harmless disagreement is converted into intervention cost"
        elif row["family"] == "controller_instability":
            reason = "direct recovery competes with taxonomy overhead"
        elif row["family"] in {"model_misspecification", "out_of_distribution_action"}:
            reason = "robust fallback captures much of the value without classification"
        elif row["family"] == "goal_specification_ambiguity":
            reason = "goal ambiguity benefits from probing only when label value exceeds cost"
        else:
            reason = "diagnostic action cost exceeds the realized value of the label"
        cases.append(
            {
                "case_id": index,
                "split": row["split"],
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


def terminal_decision(
    hard_metrics: list[dict[str, object]],
    hard_pairwise: list[dict[str, object]],
    ablation_metrics: list[dict[str, object]],
    stress_metrics: list[dict[str, object]],
    fixed_metrics: list[dict[str, object]],
) -> dict[str, object]:
    by_method = {row["method"]: row for row in hard_metrics}
    proposed = by_method[PROPOSED_V5]
    references = {name: by_method[name] for name in REFERENCE_METHODS if name in by_method}
    best_success_name, best_success = max(references.items(), key=lambda item: float(item[1]["task_success"]))
    best_safety_name, best_safety = min(references.items(), key=lambda item: float(item[1]["unsafe_commit_rate"]))
    best_regret_name, best_regret = min(references.items(), key=lambda item: float(item[1]["planning_regret_to_oracle"]))
    best_utility_name, best_utility = max(references.items(), key=lambda item: float(item[1]["robust_utility"]))
    best_recall_name, best_recall = max(references.items(), key=lambda item: float(item[1]["useful_disagreement_recall"]))

    pair_lookup = {(row["baseline"], row["metric"]): row for row in hard_pairwise}
    success_pair = pair_lookup[(best_success_name, "task_success")]
    regret_pair = pair_lookup[(best_regret_name, "planning_regret_to_oracle")]
    utility_pair = pair_lookup[(best_utility_name, "robust_utility")]

    success_gate = success_pair["winner"] == PROPOSED_V5
    safety_gate = (
        float(proposed["unsafe_commit_rate"]) <= float(best_safety["unsafe_commit_rate"]) + 0.005
        and float(proposed["task_success"]) >= float(best_safety["task_success"]) - 0.025
    )
    regret_gate = regret_pair["winner"] == PROPOSED_V5 or (
        regret_pair["winner"] == "statistical_tie" and float(proposed["task_success"]) > float(by_method[best_regret_name]["task_success"])
    )
    utility_gate = utility_pair["winner"] == PROPOSED_V5 or float(proposed["robust_utility"]) >= float(best_utility["robust_utility"]) - 0.015
    diagnosis_gate = (
        float(proposed["disagreement_family_accuracy"]) >= 0.60
        and float(proposed["useful_disagreement_recall"]) >= float(best_recall["useful_disagreement_recall"]) - 0.020
    )
    false_alarm_gate = float(proposed["noise_false_alarm_rate"]) <= 0.28

    ablation_by_name = {row["method"]: row for row in ablation_metrics}
    full = ablation_by_name["full_risk_bounded_typed_disagreement_protocol_v5"]
    ablation_beats = [
        name
        for name, row in ablation_by_name.items()
        if name != "full_risk_bounded_typed_disagreement_protocol_v5"
        and (
            float(row["task_success"]) >= float(full["task_success"]) - 0.002
            or float(row["planning_regret_to_oracle"]) <= float(full["planning_regret_to_oracle"]) + 0.002
            or float(row["robust_utility"]) >= float(full["robust_utility"]) - 0.002
        )
    ]
    ablation_gate = len(ablation_beats) == 0

    max_stress = max(float(row["stress_level"]) for row in stress_metrics)
    max_stress_rows = [row for row in stress_metrics if abs(float(row["stress_level"]) - max_stress) < 1e-9 and row["method"] != ORACLE]
    stress_proposed = [row for row in max_stress_rows if row["method"] == PROPOSED_V5][0]
    stress_best = max([row for row in max_stress_rows if row["method"] != PROPOSED_V5], key=lambda row: float(row["robust_utility"]))
    stress_gate = (
        float(stress_proposed["robust_utility"]) >= float(stress_best["robust_utility"]) - 0.020
        and float(stress_proposed["task_success"]) >= float(stress_best["task_success"]) - 0.035
    )

    budget_rows = [row for row in fixed_metrics if abs(float(row["risk_budget"]) - 0.05) < 1e-9]
    budget_proposed = [row for row in budget_rows if row["method"] == PROPOSED_V5][0]
    budget_best = max([row for row in budget_rows if row["method"] != PROPOSED_V5], key=lambda row: float(row["task_success"]))
    fixed_risk_gate = (
        float(budget_proposed["unsafe_commit_rate"]) <= 0.05
        and float(budget_proposed["deployment_coverage"]) > 0.05
        and float(budget_proposed["task_success"]) >= float(budget_best["task_success"]) - 0.035
    )
    scope_gate = False

    empirical_gates = [
        success_gate,
        safety_gate,
        regret_gate,
        utility_gate,
        diagnosis_gate,
        false_alarm_gate,
        ablation_gate,
        stress_gate,
        fixed_risk_gate,
    ]
    terminal = "STRONG_REVISE" if all(empirical_gates) else "KILL_ARCHIVE"
    reasons = []
    if not success_gate:
        reasons.append(
            f"v5 hard success {float(proposed['task_success']):.5f} does not beat {best_success_name} {float(best_success['task_success']):.5f}"
        )
    if not safety_gate:
        reasons.append(
            f"v5 unsafe {float(proposed['unsafe_commit_rate']):.5f} is not a useful improvement over safest {best_safety_name} {float(best_safety['unsafe_commit_rate']):.5f}"
        )
    if not regret_gate:
        reasons.append(
            f"v5 regret {float(proposed['planning_regret_to_oracle']):.5f} trails {best_regret_name} {float(best_regret['planning_regret_to_oracle']):.5f}"
        )
    if not utility_gate:
        reasons.append(
            f"v5 utility {float(proposed['robust_utility']):.5f} trails {best_utility_name} {float(best_utility['robust_utility']):.5f}"
        )
    if not diagnosis_gate:
        reasons.append("v5 diagnosis accuracy/recall does not clear the frozen diagnostic gate")
    if not false_alarm_gate:
        reasons.append(f"v5 false-alarm rate {float(proposed['noise_false_alarm_rate']):.5f} exceeds the frozen gate")
    if not ablation_gate:
        reasons.append("ablations match or beat the full mechanism: " + ", ".join(ablation_beats))
    if not stress_gate:
        reasons.append(
            f"maximum-stress utility/success is dominated by {stress_best['method']}"
        )
    if not fixed_risk_gate:
        reasons.append(
            f"fixed-risk budget 0.05 is dominated by {budget_best['method']} or has insufficient coverage"
        )
    if not scope_gate:
        reasons.append("scope gate fails because no real robot, accepted high-fidelity benchmark, or trained checkpoint evidence exists")
    if not reasons:
        reasons.append("all frozen empirical gates passed, but scope still requires external validation")

    return {
        "terminal": terminal,
        "iclr_main_ready": False,
        "best_success_reference": best_success_name,
        "best_safety_reference": best_safety_name,
        "best_regret_reference": best_regret_name,
        "best_utility_reference": best_utility_name,
        "best_recall_reference": best_recall_name,
        "max_stress_reference": stress_best["method"],
        "fixed_risk_reference": budget_best["method"],
        "success_gate": success_gate,
        "safety_gate": safety_gate,
        "regret_gate": regret_gate,
        "utility_gate": utility_gate,
        "diagnosis_gate": diagnosis_gate,
        "false_alarm_gate": false_alarm_gate,
        "ablation_gate": ablation_gate,
        "stress_gate": stress_gate,
        "fixed_risk_gate": fixed_risk_gate,
        "scope_gate": scope_gate,
        "ablation_beats": ablation_beats,
        "reasons": reasons,
    }


def write_dataset_summary() -> int:
    fieldnames = [
        "seed",
        "split",
        "task",
        "family",
        "episode",
        "true_risk",
        "variance_signal",
        "label_value",
    ] + FEATURE_NAMES
    count = 0
    handle, writer = open_writer(RESULTS / "dataset_summary.csv", fieldnames)
    try:
        for seed in SEEDS:
            for split in SPLITS:
                for task in TASKS:
                    for family in FAMILIES:
                        features, true_risk, variance_signal, label_value = generate_dataset(seed, split, task, family, EPISODES_PER_CELL)
                        for episode in range(EPISODES_PER_CELL):
                            row = {
                                "seed": seed,
                                "split": split.name,
                                "task": task.name,
                                "family": family.name,
                                "episode": episode,
                                "true_risk": float(true_risk[episode]),
                                "variance_signal": float(variance_signal[episode]),
                                "label_value": float(label_value[episode]),
                            }
                            for index, feature_name in enumerate(FEATURE_NAMES):
                                row[feature_name] = float(features[episode, index])
                            writer.writerow({key: fmt(value) for key, value in row.items()})
                            count += 1
    finally:
        handle.close()
    return count


def run_main_benchmark() -> tuple[list[dict[str, object]], int]:
    fieldnames = [
        "seed",
        "split",
        "task",
        "family",
        "method",
        "episode",
        "stress_level",
        "risk_budget",
        "action",
        "predicted_family",
        "true_risk",
        "predicted_risk",
        "variance_signal",
        "label_value",
    ] + METRICS
    group_rows = []
    count = 0
    handle, writer = open_writer(RESULTS / "rollouts.csv", fieldnames)
    try:
        for split in SPLITS:
            for method in METHODS:
                for seed in SEEDS:
                    for task in TASKS:
                        for family in FAMILIES:
                            rows, group = simulate_group(seed, split, task, family, method, EPISODES_PER_CELL)
                            for row in rows:
                                writer.writerow({key: fmt(value) for key, value in row.items()})
                                count += 1
                            group_rows.append(group)
    finally:
        handle.close()
    return group_rows, count


def run_ablation_benchmark() -> tuple[list[dict[str, object]], int]:
    fieldnames = [
        "seed",
        "split",
        "task",
        "family",
        "method",
        "episode",
        "stress_level",
        "risk_budget",
        "action",
        "predicted_family",
        "true_risk",
        "predicted_risk",
        "variance_signal",
        "label_value",
    ] + METRICS
    hard_splits = [split for split in SPLITS if split.name in HARD_SPLITS]
    group_rows = []
    count = 0
    handle, writer = open_writer(RESULTS / "ablation_rollouts.csv", fieldnames)
    try:
        method = METHOD_BY_NAME[PROPOSED_V5]
        for ablation in ABLATIONS:
            for split in hard_splits:
                for seed in SEEDS:
                    for task in TASKS:
                        for family in FAMILIES:
                            rows, group = simulate_group(seed, split, task, family, method, EPISODES_PER_CELL, ablation=ablation)
                            for row in rows:
                                writer.writerow({key: fmt(value) for key, value in row.items()})
                                count += 1
                            group_rows.append(group)
    finally:
        handle.close()
    return group_rows, count


def run_stress_sweep() -> tuple[list[dict[str, object]], int]:
    fieldnames = [
        "seed",
        "split",
        "task",
        "family",
        "method",
        "episode",
        "stress_level",
        "risk_budget",
        "action",
        "predicted_family",
        "true_risk",
        "predicted_risk",
        "variance_signal",
        "label_value",
    ] + METRICS
    split = SPLIT_BY_NAME["combined_disagreement_stress"]
    group_rows = []
    count = 0
    handle, writer = open_writer(RESULTS / "stress_sweep_raw.csv", fieldnames)
    try:
        for stress in STRESS_LEVELS:
            for method_name in STRESS_METHODS:
                method = METHOD_BY_NAME[method_name]
                for seed in SEEDS:
                    for task in TASKS:
                        for family in FAMILIES:
                            rows, group = simulate_group(seed, split, task, family, method, EPISODES_PER_CELL, stress_override=stress)
                            for row in rows:
                                writer.writerow({key: fmt(value) for key, value in row.items()})
                                count += 1
                            group_rows.append(group)
    finally:
        handle.close()
    return group_rows, count


def run_fixed_risk_benchmark() -> tuple[list[dict[str, object]], int]:
    fieldnames = [
        "seed",
        "split",
        "task",
        "family",
        "method",
        "episode",
        "stress_level",
        "risk_budget",
        "action",
        "predicted_family",
        "true_risk",
        "predicted_risk",
        "variance_signal",
        "label_value",
    ] + METRICS
    split = SPLIT_BY_NAME["rare_adversarial_shift"]
    group_rows = []
    count = 0
    handle, writer = open_writer(RESULTS / "fixed_risk_raw.csv", fieldnames)
    try:
        for budget in RISK_BUDGETS:
            for method_name in FIXED_RISK_METHODS:
                method = METHOD_BY_NAME[method_name]
                for seed in SEEDS:
                    for task in TASKS:
                        for family in FAMILIES:
                            rows, group = simulate_group(
                                seed,
                                split,
                                task,
                                family,
                                method,
                                EPISODES_PER_CELL,
                                deployment_budget=budget,
                            )
                            for row in rows:
                                writer.writerow({key: fmt(value) for key, value in row.items()})
                                count += 1
                            group_rows.append(group)
    finally:
        handle.close()
    return group_rows, count


def write_summary(
    row_counts: dict[str, int],
    hard_metrics: list[dict[str, object]],
    hard_pairwise: list[dict[str, object]],
    ablation_metrics: list[dict[str, object]],
    stress_metrics: list[dict[str, object]],
    fixed_metrics: list[dict[str, object]],
    negative_cases: list[dict[str, object]],
    decision: dict[str, object],
) -> None:
    by_method = {row["method"]: row for row in hard_metrics}
    proposed = by_method[PROPOSED_V5]
    best_success = by_method[decision["best_success_reference"]]
    best_safety = by_method[decision["best_safety_reference"]]
    best_regret = by_method[decision["best_regret_reference"]]
    best_utility = by_method[decision["best_utility_reference"]]
    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("Paper 97: model_disagreement_protocols expanded v5 evidence audit\n")
        handle.write(f"Terminal decision: {decision['terminal']}\n")
        handle.write("ICLR main ready: no\n")
        handle.write(
            "Design: 6 robotics tasks x 8 disagreement families x 8 splits x 14 methods, "
            f"{len(SEEDS)} seeds, {EPISODES_PER_CELL} episodes per seed/task/family/split/method cell.\n"
        )
        handle.write("Claim under test: typed disagreement should choose commit/probe/switch/abstain/recover better than direct robust or recovery baselines.\n\n")
        handle.write("Row counts:\n")
        for key, value in row_counts.items():
            handle.write(f"- {key}: {value}\n")
        handle.write("\nHard-aggregate evidence:\n")
        for row in sorted(hard_metrics, key=lambda item: float(item["task_success"]), reverse=True):
            handle.write(
                f"- {row['method']}: success={float(row['task_success']):.5f} +/- {float(row['ci95_task_success']):.5f}, "
                f"unsafe={float(row['unsafe_commit_rate']):.5f}, type_acc={float(row['disagreement_family_accuracy']):.5f}, "
                f"recall={float(row['useful_disagreement_recall']):.5f}, false_alarm={float(row['noise_false_alarm_rate']):.5f}, "
                f"regret={float(row['planning_regret_to_oracle']):.5f}, utility={float(row['robust_utility']):.5f}\n"
            )
        handle.write("\nReference winners:\n")
        handle.write(f"- best_success_reference={decision['best_success_reference']}; v5={float(proposed['task_success']):.5f}; best={float(best_success['task_success']):.5f}\n")
        handle.write(f"- best_safety_reference={decision['best_safety_reference']}; v5={float(proposed['unsafe_commit_rate']):.5f}; best={float(best_safety['unsafe_commit_rate']):.5f}\n")
        handle.write(f"- best_regret_reference={decision['best_regret_reference']}; v5={float(proposed['planning_regret_to_oracle']):.5f}; best={float(best_regret['planning_regret_to_oracle']):.5f}\n")
        handle.write(f"- best_utility_reference={decision['best_utility_reference']}; v5={float(proposed['robust_utility']):.5f}; best={float(best_utility['robust_utility']):.5f}\n")
        handle.write(f"- best_recall_reference={decision['best_recall_reference']}\n")
        handle.write(f"- max_stress_reference={decision['max_stress_reference']}\n")
        handle.write(f"- fixed_risk_reference={decision['fixed_risk_reference']}\n")
        handle.write("\nGate outcomes:\n")
        for key in [
            "success_gate",
            "safety_gate",
            "regret_gate",
            "utility_gate",
            "diagnosis_gate",
            "false_alarm_gate",
            "ablation_gate",
            "stress_gate",
            "fixed_risk_gate",
            "scope_gate",
        ]:
            handle.write(f"- {key}: {decision[key]}\n")
        handle.write("\nTerminal rationale:\n")
        for reason in decision["reasons"]:
            handle.write(f"- {reason}\n")
        handle.write("\nAblation summary:\n")
        for row in sorted(ablation_metrics, key=lambda item: float(item["robust_utility"]), reverse=True):
            handle.write(
                f"- {row['method']}: success={float(row['task_success']):.5f}, regret={float(row['planning_regret_to_oracle']):.5f}, "
                f"utility={float(row['robust_utility']):.5f}, false_alarm={float(row['noise_false_alarm_rate']):.5f}\n"
            )
        handle.write("\nRepresentative negative cases:\n")
        for row in negative_cases[:8]:
            handle.write(
                f"- {row['split']} / {row['task']} / {row['family']} seed {row['seed']}: "
                f"success={float(row['task_success']):.5f}, unsafe={float(row['unsafe_commit_rate']):.5f}, "
                f"regret={float(row['planning_regret_to_oracle']):.5f}; {row['failure_mode']}\n"
            )
        handle.write(f"\nPairwise rows: {len(hard_pairwise)}\n")
        handle.write("No hardware validation is claimed; this is a local CPU-only executable surrogate audit.\n")
        handle.write(f"terminal={decision['terminal']}\n")


def main() -> None:
    dataset_rows = write_dataset_summary()
    main_group_rows, main_rollout_rows = run_main_benchmark()
    seed_metrics = aggregate(main_group_rows, ["seed", "method", "split"])
    metrics = aggregate(seed_metrics, ["method", "split"])
    per_task_family = aggregate(main_group_rows, ["method", "split", "task", "family"])
    hard_group_rows = [row for row in main_group_rows if row["split"] in HARD_SPLITS]
    hard_seed_metrics = aggregate(hard_group_rows, ["seed", "method"])
    hard_metrics = aggregate(hard_seed_metrics, ["method"])
    hard_pairwise = pairwise_seed_tests(hard_seed_metrics, PROPOSED_V5, REFERENCE_METHODS)

    ablation_group_rows, ablation_rollout_rows = run_ablation_benchmark()
    ablation_seed_metrics = aggregate(ablation_group_rows, ["seed", "method"])
    ablation_metrics = aggregate(ablation_seed_metrics, ["method"])

    stress_group_rows, stress_rollout_rows = run_stress_sweep()
    stress_seed_metrics = aggregate(stress_group_rows, ["seed", "method", "stress_level"])
    stress_metrics = aggregate(stress_seed_metrics, ["method", "stress_level"])

    fixed_group_rows, fixed_risk_rows = run_fixed_risk_benchmark()
    fixed_seed_metrics = aggregate(fixed_group_rows, ["seed", "method", "risk_budget"])
    fixed_metrics = aggregate(fixed_seed_metrics, ["method", "risk_budget"])
    fixed_pairwise = pairwise_seed_tests(
        [row for row in fixed_seed_metrics if abs(float(row["risk_budget"]) - 0.05) < 1e-9],
        PROPOSED_V5,
        [name for name in FIXED_RISK_METHODS if name != PROPOSED_V5],
    )

    negative_cases = build_negative_cases(main_group_rows)
    decision = terminal_decision(hard_metrics, hard_pairwise, ablation_metrics, stress_metrics, fixed_metrics)

    write_csv(RESULTS / "main_group_metrics.csv", main_group_rows)
    write_csv(RESULTS / "seed_task_family_metrics.csv", main_group_rows)
    write_csv(RESULTS / "main_seed_metrics.csv", seed_metrics)
    write_csv(RESULTS / "metrics.csv", metrics)
    write_csv(RESULTS / "per_task_family_metrics.csv", per_task_family)
    write_csv(RESULTS / "hard_aggregate_seed_metrics.csv", hard_seed_metrics)
    write_csv(RESULTS / "hard_aggregate_metrics.csv", hard_metrics)
    write_csv(RESULTS / "pairwise_stats.csv", hard_pairwise)
    write_csv(RESULTS / "ablation_seed_metrics.csv", ablation_seed_metrics)
    write_csv(RESULTS / "ablation_metrics.csv", ablation_metrics)
    write_csv(RESULTS / "stress_sweep_seed_metrics.csv", stress_seed_metrics)
    write_csv(RESULTS / "stress_sweep.csv", stress_metrics)
    write_csv(RESULTS / "fixed_risk_seed_metrics.csv", fixed_seed_metrics)
    write_csv(RESULTS / "fixed_risk_metrics.csv", fixed_metrics)
    write_csv(RESULTS / "fixed_risk_pairwise_stats.csv", fixed_pairwise)
    write_csv(RESULTS / "failure_cases.csv", negative_cases)

    hard_sorted = sorted([row for row in hard_metrics if row["method"] != ORACLE], key=lambda row: float(row["task_success"]), reverse=True)
    make_latex_table(
        RESULTS / "hard_aggregate_table.tex",
        hard_sorted,
        [
            ("method", "Method"),
            ("task_success", "Success"),
            ("unsafe_commit_rate", "Unsafe"),
            ("useful_disagreement_recall", "Recall"),
            ("noise_false_alarm_rate", "False alarm"),
            ("planning_regret_to_oracle", "Regret"),
            ("robust_utility", "Utility"),
        ],
        limit=14,
    )
    combined_sorted = sorted(
        [row for row in metrics if row["split"] == "combined_disagreement_stress" and row["method"] != ORACLE],
        key=lambda row: float(row["task_success"]),
        reverse=True,
    )
    make_latex_table(
        RESULTS / "combined_stress_table.tex",
        combined_sorted,
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
    make_latex_table(
        RESULTS / "pairwise_decision_table.tex",
        [row for row in hard_pairwise if row["baseline"] in {decision["best_success_reference"], decision["best_regret_reference"], decision["best_utility_reference"]}],
        [
            ("baseline", "Baseline"),
            ("metric", "Metric"),
            ("proposed_mean", "V5"),
            ("baseline_mean", "Baseline"),
            ("mean_diff_proposed_minus_baseline", "Diff"),
            ("ci95_diff", "CI95"),
            ("winner", "Winner"),
        ],
        limit=24,
    )
    make_latex_table(
        RESULTS / "ablation_table.tex",
        sorted(ablation_metrics, key=lambda row: float(row["robust_utility"]), reverse=True),
        [
            ("method", "Ablation"),
            ("task_success", "Success"),
            ("planning_regret_to_oracle", "Regret"),
            ("robust_utility", "Utility"),
            ("noise_false_alarm_rate", "False alarm"),
        ],
    )
    make_latex_table(
        RESULTS / "stress_table.tex",
        [row for row in stress_metrics if abs(float(row["stress_level"]) - 1.0) < 1e-9 and row["method"] != ORACLE],
        [
            ("method", "Method"),
            ("task_success", "Success"),
            ("unsafe_commit_rate", "Unsafe"),
            ("planning_regret_to_oracle", "Regret"),
            ("robust_utility", "Utility"),
        ],
    )
    make_latex_table(
        RESULTS / "fixed_risk_table.tex",
        [row for row in fixed_metrics if abs(float(row["risk_budget"]) - 0.05) < 1e-9],
        [
            ("method", "Method"),
            ("task_success", "Success"),
            ("unsafe_commit_rate", "Unsafe"),
            ("deployment_coverage", "Coverage"),
            ("robust_utility", "Utility"),
        ],
    )
    make_latex_table(
        RESULTS / "negative_cases_table.tex",
        negative_cases,
        [
            ("split", "Split"),
            ("task", "Task"),
            ("family", "Family"),
            ("task_success", "Success"),
            ("planning_regret_to_oracle", "Regret"),
            ("failure_mode", "Failure"),
        ],
        limit=8,
    )

    row_counts = {
        "dataset_summary_rows": dataset_rows,
        "main_rollout_rows": main_rollout_rows,
        "main_group_rows": len(main_group_rows),
        "main_seed_metric_rows": len(seed_metrics),
        "main_metric_rows": len(metrics),
        "hard_seed_rows": len(hard_seed_metrics),
        "hard_metric_rows": len(hard_metrics),
        "hard_pairwise_rows": len(hard_pairwise),
        "ablation_rollout_rows": ablation_rollout_rows,
        "ablation_seed_rows": len(ablation_seed_metrics),
        "ablation_metric_rows": len(ablation_metrics),
        "stress_rollout_rows": stress_rollout_rows,
        "stress_seed_rows": len(stress_seed_metrics),
        "stress_metric_rows": len(stress_metrics),
        "fixed_risk_rows": fixed_risk_rows,
        "fixed_risk_seed_rows": len(fixed_seed_metrics),
        "fixed_risk_metric_rows": len(fixed_metrics),
        "fixed_risk_pairwise_rows": len(fixed_pairwise),
        "negative_cases": len(negative_cases),
    }

    plot_figures(hard_metrics, ablation_metrics, stress_metrics, fixed_metrics)
    write_summary(row_counts, hard_metrics, hard_pairwise, ablation_metrics, stress_metrics, fixed_metrics, negative_cases, decision)

    print(f"Paper 97 expanded v5 evidence audit complete: {decision['terminal']}")
    print("ICLR main ready: no")
    print("Reasons:")
    for reason in decision["reasons"]:
        print("-", reason)
    print("Wrote results to", RESULTS)


if __name__ == "__main__":
    main()
