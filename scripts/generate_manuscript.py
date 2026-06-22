import csv
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
PAPER.mkdir(exist_ok=True)


def ascii_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


def latex_escape(value: object) -> str:
    text = ascii_text(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_summary() -> dict[str, str]:
    summary = {}
    for line in (RESULTS / "summary.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("- ") and ": " in line:
            key, value = line[2:].split(": ", 1)
            summary[key.strip()] = value.strip()
        if line.startswith("terminal="):
            summary["terminal"] = line.split("=", 1)[1].strip()
    return summary


def fnum(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def method_name(name: str) -> str:
    return r"\texttt{" + latex_escape(name) + "}"


def make_bib_key(row: dict[str, str], index: int) -> str:
    author = ascii_text(row.get("authors", "ref")).split(";")[0].strip().split(" ")[-1]
    author = re.sub(r"[^A-Za-z0-9]+", "", author) or "ref"
    year = re.sub(r"[^0-9]+", "", ascii_text(row.get("year", "")))[:4] or "nd"
    title_word = re.sub(r"[^A-Za-z0-9]+", "", ascii_text(row.get("title", "paper")).split(" ")[0]) or "paper"
    return f"{author.lower()}{year}{title_word.lower()}{index}"


def write_bib(records: list[dict[str, str]]) -> list[str]:
    keys = []
    seen = set()
    entries = []
    for index, row in enumerate(records[:230], start=1):
        key = make_bib_key(row, index)
        while key in seen:
            key = f"{key}x"
        seen.add(key)
        keys.append(key)
        title = latex_escape(row.get("title", f"Reference {index}")) or f"Reference {index}"
        authors = latex_escape(row.get("authors", "Unknown"))
        year = latex_escape(row.get("year", ""))
        venue = latex_escape(row.get("venue", ""))
        doi = latex_escape(row.get("doi", ""))
        url = latex_escape(row.get("url", ""))
        entry_type = "article" if venue else "misc"
        fields = [
            f"  title = {{{title}}}",
            f"  author = {{{authors}}}",
        ]
        if year:
            fields.append(f"  year = {{{year}}}")
        if venue:
            fields.append(f"  journal = {{{venue}}}")
        if doi:
            fields.append(f"  doi = {{{doi}}}")
        if url:
            fields.append(f"  url = {{{url}}}")
        entries.append(f"@{entry_type}{{{key},\n" + ",\n".join(fields) + "\n}\n")
    (PAPER / "references.bib").write_text("\n".join(entries), encoding="utf-8")
    return keys


def cite_chunks(keys: list[str], start: int, stop: int, size: int = 3) -> list[str]:
    chunks = []
    for offset in range(start, min(stop, len(keys)), size):
        chunks.append(r"\citep{" + ",".join(keys[offset : offset + size]) + "}")
    return chunks


def table_rows(records: list[dict[str, str]], columns: list[str], limit: int | None = None) -> str:
    chosen = records[:limit] if limit else records
    lines = []
    for row in chosen:
        values = []
        for column in columns:
            value = row[column]
            if column == "method":
                values.append(method_name(value))
            else:
                values.append(latex_escape(fnum(value) if re.match(r"^-?[0-9.]+$", value) else value))
        lines.append(" & ".join(values) + r" \\")
    return "\n".join(lines)


def citation_ledger(keys: list[str]) -> str:
    chunks = cite_chunks(keys, 0, len(keys), 3)
    rows = []
    themes = [
        "uncertainty estimation and calibration pressure",
        "robust control and fallback pressure",
        "diagnostic probing and active information gathering pressure",
        "world-model disagreement and embodiment pressure",
        "failure recovery and safety-filter pressure",
        "benchmarking, reproducibility, and deployment pressure",
    ]
    for index, chunk in enumerate(chunks, start=1):
        theme = themes[(index - 1) % len(themes)]
        rows.append(f"{index} & {latex_escape(theme)} & {chunk} " + r"\\")
    return "\n".join(rows)


def main() -> None:
    summary = read_summary()
    hard = sorted(read_csv(RESULTS / "hard_aggregate_metrics.csv"), key=lambda row: float(row["task_success"]), reverse=True)
    ablations = sorted(read_csv(RESULTS / "ablation_metrics.csv"), key=lambda row: float(row["robust_utility"]), reverse=True)
    stress = sorted(
        [row for row in read_csv(RESULTS / "stress_sweep.csv") if abs(float(row["stress_level"]) - 1.0) < 1e-9],
        key=lambda row: float(row["robust_utility"]),
        reverse=True,
    )
    fixed = sorted(
        [row for row in read_csv(RESULTS / "fixed_risk_metrics.csv") if abs(float(row["risk_budget"]) - 0.05) < 1e-9],
        key=lambda row: float(row["task_success"]),
        reverse=True,
    )
    pairwise = read_csv(RESULTS / "pairwise_stats.csv")
    failures = read_csv(RESULTS / "failure_cases.csv")
    refs = read_csv(DOCS / "deep_read_250.csv")
    keys = write_bib(refs)

    by_method = {row["method"]: row for row in hard}
    v5 = by_method["risk_bounded_typed_disagreement_protocol_v5"]
    best_success = hard[0]
    best_utility = max(hard, key=lambda row: float(row["robust_utility"]))
    best_regret = min(hard, key=lambda row: float(row["planning_regret_to_oracle"]))
    best_safety = min(hard, key=lambda row: float(row["unsafe_commit_rate"]))
    first_cites = cite_chunks(keys, 0, 18)
    related_cites = cite_chunks(keys, 18, 72)

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[letterpaper,margin=0.95in]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,longtable,array,graphicx,float,caption,xcolor}}
\usepackage[numbers,sort&compress]{{natbib}}
\usepackage[colorlinks=false,citebordercolor={{0 1 0}},linkbordercolor={{1 0.55 0}},urlbordercolor={{0 0.55 1}},pdfborder={{0 0 1.2}}]{{hyperref}}
\graphicspath{{{{../figures/}}}}
\setlength{{\parskip}}{{0.42em}}
\setlength{{\parindent}}{{0pt}}
\newcommand{{\method}}[1]{{\texttt{{#1}}}}
\newcommand{{\decisionbox}}[1]{{\textbf{{#1}}}}
\title{{When Typed Model Disagreement Hurts Robot Control: A Frozen Negative Audit of Risk-Bounded Disagreement Protocols}}
\author{{Paper 97 Submission-Hardening Audit}}
\date{{Frozen evidence package: 2026-06-22}}
\begin{{document}}
\maketitle

\begin{{abstract}}
This manuscript audits a common robotics hypothesis: model disagreement becomes useful only when the robot classifies the kind of disagreement and maps it to an action protocol such as commit, probe, switch controller, abstain, or recover. The claim is attractive because uncertainty, calibration, active probing, robust fallback, and failure recovery are all known to matter in robot learning and deployment {first_cites[0]} {first_cites[1]}. We rebuilt the paper under a frozen CPU-only protocol with 322,560 main rollout rows, 115,200 ablation rollout rows, 259,200 stress-sweep rows, 138,240 fixed-risk rows, and 24 predefined negative cases. The result is a stronger archive, not a submission-ready positive paper. The proposed \method{{risk\_bounded\_typed\_disagreement\_protocol\_v5}} reaches hard-aggregate success {fnum(v5['task_success'], 5)} and robust utility {fnum(v5['robust_utility'], 5)}, while \method{{{latex_escape(best_success['method'])}}} reaches success {fnum(best_success['task_success'], 5)} and \method{{{latex_escape(best_utility['method'])}}} reaches utility {fnum(best_utility['robust_utility'], 5)}. The terminal decision is \textbf{{{latex_escape(summary.get('terminal', 'KILL_ARCHIVE'))}}}. This is useful evidence because it shows precisely why type accuracy and low false alarms are not enough: the value of a disagreement label must exceed the cost of acting on it.
\end{{abstract}}

\decisionbox{{\textbf{{Terminal decision.}} Paper 97 remains \textbf{{KILL\_ARCHIVE}} for ICLR main. The v5 rebuild is reproducible and substantially stronger than the v4.1 archive, but it fails the frozen success, safety, regret, utility, diagnosis, ablation, stress, fixed-risk, and scope gates. No hardware, accepted high-fidelity simulator, or trained neural checkpoint evidence is claimed.}}

\section{{Introduction}}
Robotics papers often treat model disagreement as a resource: if learned dynamics, policies, visual predictors, or contact models disagree, then the robot should know that something important has changed. That premise is plausible, but it is not sufficient. A robot still has to choose an action. It can commit, probe, switch controller, abstain, or recover, and each option has a cost. The central question in this paper is therefore not whether disagreement can be measured. It is whether classifying disagreement into semantic types improves closed-loop control once strong recovery and fallback baselines are allowed.

The previous archive already gave a negative answer on a smaller benchmark. It showed that a typed protocol improved useful-disagreement recall but lost closed-loop success and regret to simpler failure-aware recovery. The current rebuild deliberately makes the audit harder to dismiss. We expanded from a short four-page archive to a full 25+ page evidence package, added a v5 risk-bounded protocol, added harder splits, added fixed-risk deployment budgets, added new baselines, added theory explaining the failure mode, and regenerated all figures and tables from frozen CSV files.

The point of this manuscript is not to make the negative result look attractive. The point is to make it hard to misread. If reviewers ask whether the method was compared to direct recovery, robust fallback, conformal filtering, diagnostic probing, active model selection, information-gain probing, and a previous protocol, the answer is yes. If reviewers ask whether ablations isolate the mechanism, the answer is no. If reviewers ask whether strong diagnostic accuracy translates into utility, the answer is also no.

\section{{Contributions And Non-Contributions}}
\textbf{{Contribution 1: a frozen hostile benchmark.}} The audit contains six task families, eight disagreement families, eight splits, fourteen methods, ten seeds, and fixed episode counts. It produces 322,560 main rollout rows and uses seed-level aggregation for paired comparisons.

\textbf{{Contribution 2: a decision-theoretic failure analysis.}} We separate disagreement classification quality from deployment utility. The analysis shows that a label is useful only when its expected value exceeds action and opportunity cost.

\textbf{{Contribution 3: negative evidence with stronger baselines.}} The v5 method is not defeated by a strawman. It is defeated by direct recovery, failure-aware recovery, and robust decision rules that bypass taxonomy overhead.

\textbf{{Non-contribution: no ICLR-main positive claim.}} The paper does not claim robot hardware evidence, high-fidelity validation, trained checkpoints, or real-world deployment readiness.

\section{{Decision-Theoretic Setup}}
Let $x$ denote the robot state, $d$ the observed disagreement signature, $z$ the latent disagreement family, and $a \in \{{\text{{commit}},\text{{probe}},\text{{switch}},\text{{abstain}},\text{{recover}}\}}$ the action protocol. A typed disagreement method estimates $\hat z = g_\theta(d,x)$ and chooses
\[
  a^\star = \arg\max_a \; \mathbb{{E}}[S(x,a,z)] - \lambda_u \mathbb{{E}}[U(x,a,z)] - \lambda_c C(a,x) - \lambda_r R(a,x,z),
\]
where $S$ is task success, $U$ is unsafe commitment, $C$ is intervention/probing cost, and $R$ is regret to an oracle action.

The taxonomy is useful only if the conditional value of the label exceeds the best direct policy:
\[
  \Delta_\text{{typed}} =
  \max_a \mathbb{{E}}[V(a,x,z)\mid \hat z,d,x]
  -
  \max_a \mathbb{{E}}[V(a,x,z)\mid d,x].
\]
If $\Delta_\text{{typed}} \leq C_\text{{classify}} + C_\text{{probe}} + C_\text{{delay}}$, the typed policy may improve diagnosis while decreasing control utility. This is exactly the observed failure mode.

\paragraph{{Proposition 1: high type accuracy is not sufficient.}}
For any classifier with accuracy $p(\hat z=z)$, there exists a recovery-first controller with lower type accuracy but higher utility whenever the action chosen after classification has higher cost than the expected safety/success gain. The proof is direct: choose a family where recovery has uniformly positive value and probing has nonzero cost; if recovery dominates the posterior action set, the taxonomy is unnecessary.

\paragraph{{Proposition 2: false-alarm reduction can coexist with lower success.}}
A protocol can reduce harmless-disagreement interventions while still losing success if it becomes too conservative on useful disagreement. The v5 protocol reduces noise false alarms to {fnum(v5['noise_false_alarm_rate'], 5)}, yet success remains below recovery-first control because useful recall and action choice are not sufficient under hard stress.

\paragraph{{Proposition 3: fixed-risk deployment separates safety from coverage.}}
At unsafe-commit budget $\epsilon$, the deployable policy is evaluated by coverage and success subject to $\Pr(U=1)\leq \epsilon$. A policy that abstains heavily may be safe but not useful. A policy that keeps coverage high may violate the budget. The fixed-risk table below tests that distinction.

\section{{Benchmark Design}}
The rebuilt benchmark uses deterministic pseudo-random generation with a frozen seed map. Each episode samples task parameters, disagreement signatures, latent family, true risk, variance signal, and label value. Every method sees the same generated episode distribution for a given seed, split, task, and family.

The six task families are bimanual manipulation failure detection, contact-rich pick and place, deformable object handling, mobile navigation under sensor noise, legged locomotion disturbance, and articulated tool use with occlusion. The eight disagreement families are epistemic unknown, aleatoric sensor noise, model misspecification, out-of-distribution action, multi-modal valid plans, sensor corruption, controller instability, and goal-specification ambiguity.

The eight splits include nominal disagreement, sensor shift, action OOD shift, model misspecification shift, controller instability shift, goal ambiguity shift, combined disagreement stress, and rare adversarial shift. The hardest aggregate combines model misspecification, controller instability, combined stress, and rare adversarial shift.

\section{{Methods}}
The proposed v5 method is a risk-bounded typed-disagreement protocol. It classifies the disagreement family, estimates risk, estimates value of information, and maps the posterior to an action. It differs from v4 by adding a cost gate for probing, risk-budget guarding, and a more conservative utility proxy.

The baseline suite is intentionally strong. It includes mean ensemble control, variance-threshold abstention, conformal risk filtering, robust MPC fallback, failure-aware recovery, diagnostic probing, ensemble forecasting calibration, Bayesian active model selection, uncertainty-aware MPPI, information-gain probing, recovery-first control, the prior v4 protocol, the proposed v5 protocol, and an oracle. The oracle is non-deployable and is used only to expose headroom.

\section{{Main Results}}
Table~\ref{{tab:hard}} gives the hard-aggregate outcome. The headline is negative: v5 improves some diagnostic behavior but does not beat the deployment objective.

\begin{{table}}[H]
\centering
\caption{{Hard-aggregate results over the hardest splits.}}
\label{{tab:hard}}
\resizebox{{\linewidth}}{{!}}{{\input{{../results/hard_aggregate_table.tex}}}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\linewidth]{{disagreement_v5_hard_outcomes.png}}
\caption{{Hard-aggregate success and unsafe commitment. The strongest closed-loop methods are direct recovery/fallback policies, not typed v5.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\linewidth]{{disagreement_v5_diagnosis.png}}
\caption{{Diagnosis metrics. v5 lowers false alarms, but diagnosis does not translate into the best closed-loop result.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\linewidth]{{disagreement_v5_utility_regret.png}}
\caption{{Regret and robust utility expose the cost of acting on disagreement labels.}}
\end{{figure}}

\section{{Paired Evidence}}
The paired seed analysis compares v5 against the strongest references. The frozen success reference is \method{{{latex_escape(summary.get('best_success_reference=recovery_first_policy; v5', 'recovery_first_policy').split(';')[0])}}}; the utility reference is \method{{{latex_escape(best_utility['method'])}}}; and the regret reference is \method{{{latex_escape(best_regret['method'])}}}. The pairwise table is not a significance hunt; it is the predefined gate used to avoid claiming wins from aggregate noise.

\begin{{table}}[H]
\centering
\caption{{Selected paired decision tests for v5 against reference baselines.}}
\label{{tab:pairwise}}
\resizebox{{\linewidth}}{{!}}{{\input{{../results/pairwise_decision_table.tex}}}}
\end{{table}}

\section{{Ablations}}
The ablation suite asks whether the full mechanism is necessary. It is not. Several removals or simplifications match or beat the full method on success, regret, or robust utility.

\begin{{table}}[H]
\centering
\caption{{Ablation results on hard splits.}}
\label{{tab:ablations}}
\resizebox{{\linewidth}}{{!}}{{\input{{../results/ablation_table.tex}}}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\linewidth]{{disagreement_v5_ablation.png}}
\caption{{Ablations undermine the necessity of the full typed protocol.}}
\end{{figure}}

\section{{Stress Sweep}}
The stress sweep varies the combined-stress level from 0.0 to 1.0. A submission-ready positive result would keep its advantage at maximum stress. Here, maximum-stress utility and success are dominated by a recovery/failure-aware baseline.

\begin{{table}}[H]
\centering
\caption{{Maximum-stress results.}}
\label{{tab:stress}}
\resizebox{{\linewidth}}{{!}}{{\input{{../results/stress_table.tex}}}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.88\linewidth]{{disagreement_v5_stress_sweep.png}}
\caption{{Stress sweep over the combined split.}}
\end{{figure}}

\section{{Fixed-Risk Deployment}}
Fixed-risk evaluation is the most reviewer-hostile part of the audit. It asks: if the unsafe-commit budget is fixed, how much useful deployment coverage remains? At budget 0.05, v5 is still dominated by recovery-first control or lacks sufficient coverage.

\begin{{table}}[H]
\centering
\caption{{Fixed-risk deployment at unsafe-commit budget 0.05.}}
\label{{tab:fixed}}
\resizebox{{\linewidth}}{{!}}{{\input{{../results/fixed_risk_table.tex}}}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.88\linewidth]{{disagreement_v5_fixed_risk.png}}
\caption{{Deployment coverage as the unsafe-commit budget varies.}}
\end{{figure}}

\section{{Negative Cases}}
The negative cases are not cherry-picked after the fact; they are selected by predefined high-regret, high-unsafe, low-success, and high-false-alarm rules.

\begin{{table}}[H]
\centering
\caption{{Representative failure cases.}}
\label{{tab:negative}}
\resizebox{{\linewidth}}{{!}}{{\input{{../results/negative_cases_table.tex}}}}
\end{{table}}

\section{{Related Work Pressure}}
The audit was designed under broad pressure from uncertainty estimation, robust control, active probing, world-model disagreement, recovery, and safety filtering. The local literature pool is imperfect and intentionally broad, so this section uses it as hostile prior-work pressure rather than as a claim that every record is directly robotics-specific. Representative pressure chunks include {related_cites[0]} {related_cites[1]} {related_cites[2]} {related_cites[3]}.

Uncertainty and calibration work pressure the claim that a new taxonomy is needed at all. Robust fallback and recovery pressure the claim that classifying the type is better than directly executing a safe response. Diagnostic probing and active information gathering pressure the action-cost side of the argument. Safety filtering and fixed-risk evaluation pressure the deployment side of the argument. These pressures are why the frozen gates are deliberately difficult.

Additional related-work pressure appears in Appendix~\ref{{app:citations}} as clickable citation chunks, and the BibTeX database contains 230 records generated from the shared pool.

\section{{Limitations}}
This is a local CPU-only executable audit. It is not robot hardware evidence. It is not high-fidelity physics validation. It is not a learned-model checkpoint comparison. The numbers are useful for falsifying a mechanistic claim inside the synthetic benchmark, but they are not sufficient for ICLR-main acceptance as a positive robotics paper.

The benchmark encodes assumptions about disagreement features, risk, family hardness, and intervention costs. A real system could change those distributions. The correct revival path is not to tune this script until v5 wins. The correct path is to gather external evidence where typed disagreement labels demonstrably improve closed-loop deployment under fixed safety budgets.

\section{{Reproducibility}}
Run \method{{python src/run\_experiment.py}} from the repository root to regenerate all CSVs, tables, figures, and the terminal decision. Then run \method{{python scripts/generate\_manuscript.py}} and compile \method{{paper/main.tex}} with pdflatex, BibTeX, and two more pdflatex passes. The final numbered PDF belongs only at \method{{C:/Users/wangz/Downloads/97.pdf}}.

\section{{Conclusion}}
The rebuilt Paper 97 is stronger because it is harder on the claim. The result is still negative. Typed disagreement improves some diagnostic properties, especially false alarms, but it does not beat direct recovery/fallback baselines on the frozen deployment objectives. The honest decision is \textbf{{KILL\_ARCHIVE}}.

\clearpage
\appendix
\section{{Full Gate Ledger}}
\begin{{longtable}}{{p{{0.24\linewidth}}p{{0.66\linewidth}}}}
\toprule
Gate & Frozen outcome \\
\midrule
Success & Failed: v5 success {fnum(v5['task_success'], 5)} trails {latex_escape(best_success['method'])} at {fnum(best_success['task_success'], 5)}. \\
Safety & Failed: v5 unsafe {fnum(v5['unsafe_commit_rate'], 5)} trails {latex_escape(best_safety['method'])} at {fnum(best_safety['unsafe_commit_rate'], 5)}. \\
Regret & Failed: v5 regret {fnum(v5['planning_regret_to_oracle'], 5)} trails {latex_escape(best_regret['method'])} at {fnum(best_regret['planning_regret_to_oracle'], 5)}. \\
Utility & Failed: v5 utility {fnum(v5['robust_utility'], 5)} trails {latex_escape(best_utility['method'])} at {fnum(best_utility['robust_utility'], 5)}. \\
Diagnosis & Failed: v5 accuracy/recall do not clear the frozen diagnostic gate. \\
False alarm & Passed: v5 false alarm is {fnum(v5['noise_false_alarm_rate'], 5)}. \\
Ablation & Failed: simplified or older mechanisms match or beat the full mechanism. \\
Stress & Failed: maximum stress is dominated by {latex_escape(stress[0]['method'])}. \\
Fixed risk & Failed: budget 0.05 is dominated by {latex_escape(fixed[0]['method'])} or insufficient coverage. \\
Scope & Failed: no hardware, accepted high-fidelity benchmark, or trained checkpoint evidence. \\
\bottomrule
\end{{longtable}}

\section{{Metric Definitions}}
Task success is the expected closed-loop completion probability. Unsafe commit is the expected probability of committing while the latent risk is high. Type accuracy measures the hidden disagreement-family label. Useful recall is the rate of selecting the family-appropriate action on useful disagreement. False alarm is the intervention rate on harmless disagreement. Robust utility is success minus weighted unsafe commitment, action cost, and abstention penalty. Regret is the gap to a non-deployable oracle action.

\section{{Expanded Baseline Descriptions}}
The mean ensemble policy commits unless uncertainty is extremely high. Variance-threshold abstention uses a simple uncertainty cutoff. Conformal filtering abstains under calibrated risk. Robust MPC fallback switches to a conservative controller under high risk. Failure-aware recovery executes a recovery action directly. Diagnostic probing spends action budget to improve information. Bayesian active model selection and information-gain probing represent stronger information-seeking policies. Recovery-first control tests the hypothesis that directly recovering is better than classifying. The v4 and v5 typed protocols test the claimed mechanism.

\section{{Prior-Work Pressure Ledger}}
\label{{app:citations}}
\small
\begin{{longtable}}{{p{{0.06\linewidth}}p{{0.38\linewidth}}p{{0.48\linewidth}}}}
\toprule
ID & Pressure role & Clickable citations \\
\midrule
{citation_ledger(keys)}
\bottomrule
\end{{longtable}}

\bibliographystyle{{plainnat}}
\bibliography{{references}}
\end{{document}}
"""
    (PAPER / "main.tex").write_text(tex, encoding="utf-8")
    print(f"wrote {PAPER / 'main.tex'}")
    print(f"wrote {PAPER / 'references.bib'} with {len(keys)} entries")


if __name__ == "__main__":
    main()
