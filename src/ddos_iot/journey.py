"""Read small saved research reports for the progress page; no model or dataset access."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import mean, pstdev

import pandas as pd


STUDIES = "outputs/studies"
SEEDS = {42, 123, 2024}
DEVELOPMENT_COLUMNS = ["classes", "stage", "order", "macro_f1", "std", "lower", "upper", "seeds"]
FINAL_COLUMNS = ["classes", "candidate", "seeds", "macro_f1_mean", "weighted_f1_mean", "benign_false_positive_rate_mean"]
MINORITY_COLUMNS = ["classes", "class", "support", "reference_f1_mean", "selected_f1_mean",
                    "selected_recall_mean", "recall_delta_pp"]
EXPLORATION_COLUMNS = ["classes", "phase", "candidate", "decision", "macro_f1", "seed", "split", "source"]


@dataclass(frozen=True)
class Milestone:
    title: str
    phase: str
    purpose: str
    finding: str
    evidence: tuple[str, ...]
    statuses: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return all(status in {"complete", "audit_complete_reproduction_pending"} for status in self.statuses)


@dataclass
class Journey:
    milestones: list[Milestone]
    development: pd.DataFrame
    final: pd.DataFrame
    paper: pd.DataFrame
    minority: pd.DataFrame
    timing: pd.DataFrame
    interval: tuple[float, float] | None
    recall_budget_pp: float | None
    recall_budget_pass: bool | None
    final_complete: bool
    issues: list[str]
    task_milestones: dict[int, list[Milestone]]
    exploration: pd.DataFrame
    model_families: dict[int, tuple[str, ...]]


@dataclass
class BinaryJourney:
    milestones: list[Milestone]
    f1: float | None
    best_model: str | None
    issues: list[str]
    comparisons: pd.DataFrame


class _Reports:
    def __init__(self, root: Path):
        self.root, self.issues, self.cache = root, [], {}

    def json(self, relative: str) -> dict:
        if relative in self.cache:
            return self.cache[relative]
        value = {}
        path = self.root / relative
        try:
            if path.exists():
                if path.stat().st_size > 2_000_000:
                    raise ValueError("Report exceeds the page's size limit")
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("Expected a report object")
        except (OSError, ValueError) as error:
            self.issues.append(f"{relative}: {error}")
            value = {}
        self.cache[relative] = value
        return value

    def csv(self, relative: str, columns: list[str]) -> pd.DataFrame:
        path = self.root / relative
        try:
            if not path.exists():
                return pd.DataFrame(columns=columns)
            if path.stat().st_size > 2_000_000:
                raise ValueError("Report exceeds the page's size limit")
            frame = pd.read_csv(path, nrows=5000)
            if not set(columns).issubset(frame.columns):
                raise ValueError("Required result columns are missing")
            for column in columns:
                if column not in {"candidate", "class"}:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
                    if not frame[column].map(lambda v: pd.notna(v) and math.isfinite(v)).all():
                        raise ValueError(f"Invalid numbers in {column}")
            return frame[columns].copy()
        except (OSError, ValueError, pd.errors.ParserError) as error:
            self.issues.append(f"{relative}: {error}")
            return pd.DataFrame(columns=columns)


def load_binary_journey(root: Path) -> BinaryJourney:
    """Read small binary reports and bundle existence, without opening models or CSVs."""
    reports = _Reports(root)
    base = "models/requested_model_comparison"
    names = ("random_forest", "xgboost", "hist_gradient_boosting", "lightgbm", "naive_bayes")
    preprocessing = reports.json("data/processed/preprocessing_report.json")
    prepared = bool(preprocessing) and preprocessing.get("status") not in {"running", "failed", "interrupted"}
    saved = [name for name in names if (root / base / name / "model.joblib").is_file()]
    comparison = reports.json(f"{base}/model_comparison_report.json")
    best = comparison.get("best_model", {})
    best = best if isinstance(best, dict) else {}
    name = best.get("model") if best.get("model") in names else None
    score = best.get("f1")
    valid = (name is not None and best.get("status") == "success" and
             isinstance(score, (int, float)) and not isinstance(score, bool) and
             math.isfinite(score) and 0 <= score <= 1)
    comparisons = _binary_comparisons(comparison, name if valid else None, reports.issues)
    explanations = [model for model in names
                    if (report := reports.json(f"{base}/{model}/shap/shap_report.json"))
                    and isinstance(report.get("explained_rows"), int) and not isinstance(report["explained_rows"], bool)
                    and report["explained_rows"] > 0 and isinstance(report.get("method"), str) and report["method"]]
    definitions = [
        ("Prepare data", prepared, "Preprocessing and selected features are saved.",
         "Prepare the binary train and test data.", "data/processed/preprocessing_report.json"),
        ("Train models", bool(saved), f"{len(saved)} trained {'model is' if len(saved) == 1 else 'models are'} saved and ready to reuse.",
         "Train classifiers on the prepared data.", base),
        ("Compare results", valid, f"{name} is the selected model in the saved comparison.",
         "Compare models and save the selected result.", f"{base}/model_comparison_report.json"),
        ("Explain predictions", bool(explanations), "Feature explanations are saved and ready to explore.",
         "Generate feature explanations for a saved model.", f"{base}/<model>/shap/shap_report.json"),
    ]
    milestones = [Milestone(title, "Binary workflow", pending, done if complete else pending,
                            (evidence,), ("complete" if complete else "pending",))
                  for title, complete, done, pending, evidence in definitions]
    return BinaryJourney(milestones, float(score) if valid else None, name if valid else None, reports.issues, comparisons)


def _binary_comparisons(report: dict, selected: str | None, issues: list[str]) -> pd.DataFrame:
    """Preserve recorded comparison order, including scores below earlier models."""
    columns = ["order", "model", "label", "f1", "selected"]
    labels = {"random_forest": "Random forest", "xgboost": "XGBoost",
              "hist_gradient_boosting": "Hist. boosting", "lightgbm": "LightGBM", "naive_bayes": "Naive Bayes"}
    results = report.get("models", [])
    if not isinstance(results, list):
        issues.append("Binary comparison: expected saved model rows")
        return pd.DataFrame(columns=columns)
    rows, seen = [], set()
    for index, row in enumerate(results):
        if not isinstance(row, dict) or row.get("status") != "success":
            continue
        model, score = row.get("model"), row.get("f1")
        if not isinstance(model, str) or model not in labels:
            continue
        if model in seen:
            issues.append("Binary comparison: duplicate model results; chart unavailable")
            return pd.DataFrame(columns=columns)
        seen.add(model)
        if (not isinstance(score, (int, float)) or isinstance(score, bool) or
                not math.isfinite(score) or not 0 <= score <= 1):
            issues.append(f"Binary comparison: invalid F1 for {model}")
            continue
        rows.append(dict(order=index, model=model, label=labels[model], f1=float(score), selected=model == selected))
    return pd.DataFrame(rows, columns=columns)


def _milestones(reports: _Reports) -> list[Milestone]:
    definitions = [
        ("Start with controlled pilots", "Local 46-feature track",
         "Test features, class weights and decision biases cheaply before scaling up.",
         "Both tasks completed six-candidate pilots. Sampling changed class prevalence, so these scores stay in their own track.",
         ("outputs/experiments/8class_pilot_v1/experiment_report.json", "outputs/experiments/34class_pilot_v1/experiment_report.json")),
        ("Scale the eight-class ablations", "Local 46-feature track",
         "Check whether the cheap fixes still help on the full local training source.",
         "The full eight-class comparison improved F1 over its local reference, but important attack recalls fell. It was not a paper reproduction.",
         ("outputs/experiments/8class_v1/experiment_report.json",)),
        ("Make recall part of the decision", "Local 46-feature track",
         "Check whether a higher F1 also means detecting more minority attacks.",
         "Constrained decision tuning exposed the precision–recall trade-off. A higher overall F1 alone could not choose the operating point.",
         ("outputs/experiments/8class_v1/recall_constraints_all_targets_v1/report.json",)),
        ("Audit and reconstruct the paper baseline", "Published 23-feature track",
         "Align the released subsets, model settings and software before claiming a gain.",
         "Pinned author code and software references were prepared. Their official reproduction scores were measured later, in the frozen final evaluation.",
         ("outputs/baseline_audit_v1/audit.json", f"{STUDIES}/paper_fidelity_v3/8class/report.json", f"{STUDIES}/paper_fidelity_v3/34class/report.json")),
        ("Measure the cheap fixes separately", "Published 23-feature track",
         "Compare capacity, stopping, weight powers and OOF decision tuning against fixed XGBoost.",
         "The full-capacity reference remained strong. Bias tuning could improve F1 and false alarms while losing attack recall.",
         tuple(f"{STUDIES}/{k}class_v1/{suffix}" for k in (8, 34) for suffix in ("report.json", "bias_v1/report.json"))),
        ("Test stacking and targeted specialists", "Published 23-feature track",
         "Use the remaining confusion patterns to decide where another model could help.",
         "The tested XGBoost/CatBoost stack underperformed. An expanded specialist targeting Benign, minority classes and Spoofing was more promising for 34 classes.",
         tuple(f"{STUDIES}/{k}class_v1/{stage}/report.json" for k in (8, 34) for stage in ("stack_optimizer_v2", "specialist_expanded_v2"))),
        ("Confirm the comparisons across three splits", "Development validation",
         "Keep model settings and decision vectors fixed while checking split variation.",
         "The tuned 34-class specialist improved F1 on all three development splits. Its mean recall trade-off was reviewed before the final model choice; eight-class gains were not convincing.",
         (*tuple(f"{STUDIES}/{k}class_v1/{stage}/report.json" for k in (8, 34) for stage in
                 ("confirmation_v1", "bias_comparison_v1", "specialist_expanded_comparison_v2")),
          f"{STUDIES}/34class_v1/specialist_bias_confirmation_v1/report.json")),
        ("Freeze both choices and run the official tests", "Final evaluation",
         "Evaluate the predeclared models once, with paired uncertainty and audited timing.",
         "Paper baselines were reproduced. The 34-class F1 gain held up, but the recall budget did not. Eight classes retained XGBoost without a demonstrated improvement.",
         (f"{STUDIES}/final_protocol_v1/report.json", f"{STUDIES}/final_results_v1/report.json")),
    ]
    return [Milestone(title, phase, purpose, finding, evidence,
            tuple(str(reports.json(path).get("status", "unavailable")) for path in evidence))
            for title, phase, purpose, finding, evidence in definitions]


def _development(reports: _Reports) -> pd.DataFrame:
    rows = []
    stages = (("confirmation_v1", "Reference", "paper_fixed"),
              ("bias_comparison_v1", "OOF bias", "xgb_oof_bias"),
              ("specialist_expanded_comparison_v2", "Specialist", None),
              ("recall_policy_confirmation_v1", "Tuned specialist", None))
    for count in (8, 34):
        for order, (folder, label, name) in enumerate(stages):
            if count == 8 and folder == "recall_policy_confirmation_v1":
                continue
            path = f"{STUDIES}/{count}class_v1/{folder}/report.json"
            report = reports.json(path)
            if report.get("status") != "complete":
                continue
            candidate = name or report.get("selected_candidate" if folder == "recall_policy_confirmation_v1" else "candidate")
            raw_rows = report.get("results", [])
            if not isinstance(raw_rows, list):
                reports.issues.append(f"{path}: Expected saved validation rows")
                continue
            observed = [r for r in raw_rows if isinstance(r, dict) and r.get("candidate") == candidate]
            try:
                if (len(observed) != 3 or {r["seed"] for r in observed} != SEEDS or
                    any(r.get("split") != "validation" for r in observed)):
                    raise ValueError("Three unique validation seeds are required")
                scores = [float(r["macro_f1"]) for r in observed]
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in scores):
                    raise ValueError("Invalid Macro-F1")
                average, std = mean(scores), pstdev(scores)
                rows.append(dict(classes=count, stage=label, order=order, macro_f1=average,
                                 std=std, lower=average-std, upper=average+std, seeds=3))
            except (KeyError, TypeError, ValueError) as error:
                reports.issues.append(f"{path}: {error}")
    return pd.DataFrame(rows, columns=DEVELOPMENT_COLUMNS)


def _task_milestones(reports: _Reports, milestones: list[Milestone], count: int) -> list[Milestone]:
    """Keep each task's evidence and decisions independent of the other task."""
    other = 34 if count == 8 else 8
    findings = {
        0: "Compared six feature/weight settings and OOF decision tuning on a small local sample. These scores use a different protocol from the paper reproduction.",
        3: "Aligned the released subsets, model settings and pinned software. The paper reproduction was measured in the final evaluation.",
        4: ("The full-capacity XGBoost reference remained strong. Weight and threshold changes did not establish a reliable gain."
            if count == 8 else "Threshold tuning raised F1, but attack recall could fall. F1 alone was not enough to choose the model."),
        6: ("Threshold and expanded-specialist gains were inconsistent across three matched splits. Retained XGBoost."
            if count == 8 else "The expanded specialist improved F1 across three matched splits. Recall trade-offs still needed a separate decision."),
        7: ("Retained XGBoost. The frozen official tests did not demonstrate an 8-class improvement."
            if count == 8 else "The tuned specialist's F1 gain held up on the official test, but several minority classes exceeded the recall-loss limit."),
    }
    result = []

    def append(title: str, phase: str, purpose: str, finding: str, evidence: tuple[str, ...]) -> None:
        result.append(Milestone(title, phase, purpose, finding, evidence,
            tuple(str(reports.json(path).get("status", "unavailable")) for path in evidence)))

    for index, milestone in enumerate(milestones):
        if count == 34 and index in (1, 2):
            continue  # These full local ablations were run only for eight classes.
        evidence = tuple(path for path in milestone.evidence if f"/{other}class" not in path)
        if index == 5:
            append("Compare stacking", milestone.phase, "Check whether combining models helps.",
                   "The tested XGBoost/CatBoost stack underperformed the reference and was not advanced to three-split confirmation.",
                   tuple(path for path in evidence if "stack_" in path))
            append("Add targeted specialists", milestone.phase, "Focus another model on the remaining confusions.",
                   "The expanded specialist approached the reference but did not establish a reliable improvement."
                   if count == 8 else "Expanding the specialist to include Spoofing made it a promising 34-class candidate.",
                   (f"{STUDIES}/{count}class_v1/specialist_v1/report.json",
                    *tuple(path for path in evidence if "specialist_" in path)))
            continue
        if index == 7 and count == 34:
            append("Choose a recall-aware operating point", "Development validation",
                   "Tune decision biases on training-only OOF predictions and review validation recall.",
                   "Selected the tuned specialist with at most 0.5 percentage points mean recall loss per target class and no benign-FPR increase in any validation split.",
                   (f"{STUDIES}/34class_v1/specialist_bias_v1/report.json",
                    f"{STUDIES}/34class_v1/recall_policy_confirmation_v1/report.json"))
        title = "Freeze the choice and run the official test" if index == 7 else milestone.title
        if index == 7:
            evidence += (f"{STUDIES}/{count}class_v1/test_evaluation_v1/report.json",)
        append(title, milestone.phase, milestone.purpose, findings.get(index, milestone.finding), evidence)
    return result


def _exploration(reports: _Reports) -> pd.DataFrame:
    """Read earlier one-seed trials without merging protocols or score roles."""
    rows = []
    for count in (8, 34):
        sources = [("Local pilot", f"outputs/experiments/{count}class_pilot_v1/experiment_report.json", "results", "validation", "")]
        if count == 8:
            sources += [
                ("Full local study", "outputs/experiments/8class_v1/experiment_report.json", "results", "validation", ""),
                ("Local recall tuning", "outputs/experiments/8class_v1/recall_constraints_all_targets_v1/report.json", "validation_results", "validation", ""),
            ]
        base = f"{STUDIES}/{count}class_v1"
        sources += [
            ("Published training search", f"{base}/report.json", "results", "tuning", ""),
            ("Published model trials", f"{base}/report.json", "validation_results", "validation", "Initial comparison"),
        ]
        sources += [("Published model trials", f"{base}/{folder}/report.json", "validation_results", "validation", label)
                    for folder, label in (("bias_v1", "Thresholds"), ("stack_v1", "Stack v1"),
                        ("stack_optimizer_v2", "Stack v2"), ("specialist_v1", "Specialist v1"),
                        ("specialist_expanded_v2", "Expanded specialist"), ("specialist_bias_v1", "Specialist thresholds"))]
        for phase, path, key, split, label in sources:
            report = reports.json(path)
            if report.get("status") != "complete":
                continue
            observed = report.get(key, [])
            if not isinstance(observed, list):
                reports.issues.append(f"{path}: Expected saved trial rows")
                continue
            identities = set()
            for row in observed:
                if not isinstance(row, dict) or row.get("split") != split or row.get("seed") != 42:
                    continue
                try:
                    score = float(row["macro_f1"])
                    candidate = row["candidate"]
                    decision = row.get("decision") or "Model output"
                    identity = (candidate, decision)
                    if (not isinstance(candidate, str) or not candidate or not isinstance(decision, str) or
                        not math.isfinite(score) or not 0 <= score <= 1 or identity in identities):
                        raise ValueError("Invalid or duplicate trial score")
                    identities.add(identity)
                    name = f"{label} · {candidate}" if label else candidate
                    rows.append(dict(classes=count, phase=phase, candidate=name, decision=decision,
                                     macro_f1=score, seed=42, split=split, source=path))
                except (KeyError, TypeError, ValueError) as error:
                    reports.issues.append(f"{path}: {error}")
    return pd.DataFrame(rows, columns=EXPLORATION_COLUMNS)


def load_journey(root: Path) -> Journey:
    """Load an artifact-only snapshot, including honest empty/partial checkout states."""
    reports = _Reports(root)
    model_families = {}
    for count, folder in ((8, "multiclass_8_tuned"), (34, "multiclass_34")):
        comparison = reports.json(f"models/{folder}/model_comparison_report.json")
        results = comparison.get("results", [])
        names = []
        if isinstance(results, list) and comparison.get("class_count", count) == count:
            for row in results:
                if not isinstance(row, dict):
                    continue
                name, score = row.get("model"), row.get("macro_f1")
                if (isinstance(name, str) and name in ("xgboost", "lightgbm", "catboost") and name not in names
                        and isinstance(score, (int, float)) and not isinstance(score, bool)
                        and math.isfinite(score) and 0 <= score <= 1):
                    names.append(name)
        model_families[count] = tuple(names)
    milestones = _milestones(reports)
    task_milestones = {count: _task_milestones(reports, milestones, count) for count in (8, 34)}
    development = _development(reports)
    exploration = _exploration(reports)
    final_report = reports.json(f"{STUDIES}/final_results_v1/report.json")
    final_complete = (final_report.get("status") == "complete" and
                      reports.json(f"{STUDIES}/final_protocol_v1/report.json").get("status") == "complete")
    final = pd.DataFrame(columns=FINAL_COLUMNS)
    minority = pd.DataFrame(columns=MINORITY_COLUMNS)
    paper = pd.DataFrame(columns=["classes", "macro_f1", "gpu_macro_f1"])
    interval = None
    if final_complete:
        final = reports.csv(f"{STUDIES}/final_results_v1/test_summary.csv", FINAL_COLUMNS)
        if not final.empty and (not final.seeds.eq(3).all() or final.duplicated(["classes", "candidate"]).any() or
                               not final.macro_f1_mean.between(0, 1).all()):
            reports.issues.append("Final score table has invalid seed coverage or duplicate/out-of-range results.")
            final = pd.DataFrame(columns=FINAL_COLUMNS)
        paper = reports.csv(f"{STUDIES}/final_results_v1/pinned_paper_references.csv", ["classes", "macro_f1", "gpu_macro_f1"])
        minority = reports.csv(f"{STUDIES}/final_results_v1/minority_test_metrics.csv", MINORITY_COLUMNS)
        uncertainty = final_report.get("macro_f1_uncertainty", {})
        try:
            reference = final.loc[(final.classes == 34) & (final.candidate == "paper_fixed")].iloc[0]
            selected = final.loc[(final.classes == 34) & (final.candidate != "paper_fixed")].iloc[0]
            low, high = map(float, uncertainty["delta_ci95"])
            if not (all(math.isfinite(v) for v in (low, high)) and low <= high and
                    abs(float(uncertainty["reference"]) - reference.macro_f1_mean) < 1e-9 and
                    abs(float(uncertainty["candidate"]) - selected.macro_f1_mean) < 1e-9):
                raise ValueError("Confidence interval does not match final score table")
            interval = low, high
        except (KeyError, IndexError, TypeError, ValueError) as error:
            reports.issues.append(f"Final uncertainty unavailable: {error}")
    policy = reports.json(f"{STUDIES}/34class_v1/recall_policy_confirmation_v1/selection.json").get("recall_policy", {})
    try:
        budget = float(policy["maximum_recall_loss"]) * 100
        if not math.isfinite(budget) or not 0 <= budget <= 100 or policy.get("scope") != "mean":
            raise ValueError
    except (KeyError, TypeError, ValueError):
        budget = None
    timing_report = reports.json(f"{STUDIES}/34class_v1/selected_paired_timing_v1/report.json")
    timing_rows = timing_report.get("results", []) if timing_report.get("status") == "complete" else []
    if not isinstance(timing_rows, list):
        timing_rows = []
    timing = pd.DataFrame([r for r in timing_rows if isinstance(r, dict) and
        isinstance(r.get("inference_us_median"), (float, int)) and math.isfinite(r["inference_us_median"]) and r["inference_us_median"] > 0])
    flag = final_report.get("selected_34_test_recall_budget_pass") if final_complete else None
    return Journey(milestones, development, final, paper, minority, timing, interval,
                   budget, flag if isinstance(flag, bool) else None, final_complete, reports.issues,
                   task_milestones, exploration, model_families)
