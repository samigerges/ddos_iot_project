"""Offline, non-executing inspection of the versioned author source and local run."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

from .storage import file_digest, write_report, write_table

AUTHOR_REPOSITORY = "https://github.com/sa1ah-ai/iot-intrusion-detection-gradient-boosting"
PAPER_URL = "https://www.nature.com/articles/s41598-026-47399-5"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_author_script(path: Path) -> dict:
    """Parse literals and source locations only. Never import or execute the source."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    dictionaries, defaults, functions = {}, {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = {"line": node.lineno, "end_line": node.end_lineno}
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("xgb_params_all", "xgb_params_selected", "fs_params"):
                    values = {}
                    for key, value in zip(node.value.keys, node.value.values):
                        try:
                            values[ast.literal_eval(key)] = ast.literal_eval(value)
                        except ValueError:
                            values[ast.literal_eval(key)] = {"source_expression": ast.unparse(value)}
                    dictionaries[target.id] = {"line": node.lineno, "values": values}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant):
                for keyword in node.keywords:
                    if keyword.arg == "default":
                        try:
                            defaults[node.args[0].value] = ast.literal_eval(keyword.value)
                        except ValueError:
                            pass
    if not {"xgb_params_all", "xgb_params_selected", "fs_params"}.issubset(dictionaries):
        raise ValueError(f"Expected author settings are missing from {path}.")
    return {"path": str(path), "sha256": file_digest(path), "dictionaries": dictionaries,
            "argument_defaults": defaults, "functions": functions, "executed": False}


def audit_baseline(run_path: Path, sources: Path, output_path: Path) -> dict:
    manifest = _read(sources / "author_manifest.json")
    for item in manifest["files"]:
        if file_digest(sources / "author_v3" / item["path"]) != item["sha256"]:
            raise ValueError("Downloaded author source changed since its pinned manifest.")
    settings = {task: inspect_author_script(sources / "author_v3/experiment_scripts" / f"{task}_classification_experiment.py")
                for task in ("8class", "34class")}
    config = _read(run_path / "config.json")
    experiment = _read(run_path / "experiment_report.json")
    data = _read(run_path / "data_report.json")
    name = config["reference_candidate"] or config["candidates"][0]["name"]
    training = _read(run_path / f"seed_{config['seeds'][0]}" / name / "training.json")
    eight = settings["8class"]["dictionaries"]
    differences = []

    def compare(aspect: str, author: str, local: str, impact: str) -> None:
        differences.append({"aspect": aspect, "author_v3": author, "local_run": local,
                            "interpretation": impact})

    compare("source and sampling", "Raw part-*.csv pool; target 5,000,000; minimum 1,000 per original label before family mapping",
            f"Existing training partition: {data['valid_rows'] if 'valid_rows' in data else sum(data['selected_counts'].values()):,} flows; no undersampling",
            "Different cohorts and minority representation; the two Macro-F1 values are not directly comparable.")
    compare("split", "One 80/20 split stratified by task labels after undersampling",
            "Existing external train/test partition; further 80/20 raw-subtype-stratified development/validation split",
            "Local validation is a separate benchmark, not the published held-out test.")
    compare("XGBoost depth, 8 classes", str(eight["xgb_params_selected"]["values"]["max_depth"]),
            str(config["max_depth"]), "A material model-capacity difference.")
    compare("XGBoost rounds", str(eight["xgb_params_selected"]["values"]["n_estimators"]) + "; fixed, no XGBoost early stopping configured",
            f"Cap {config['n_estimators']}; inner early stopping {config['early_stopping_rounds']}; reference refit {training['best_rounds']}",
            "The 500-round cap does not imply that 500 rounds were fitted locally.")
    compare("XGBoost sampling/bins", "subsample, colsample_bytree, max_bin not set in author script (library defaults)",
            "subsample=0.9, colsample_bytree=0.9, max_bin=128", "Additional differences beyond depth/rounds.")
    compare("feature selector", "Unweighted LightGBM: 100 rounds, 31 leaves, min_data_in_leaf=100; 200,000-row random subset",
            "Inverse-weighted LightGBM: 50 estimators, 15 leaves, max_bin=63; class-preserving 200,000-row sample",
            "Both select gains above the median, but equal feature counts do not establish equal feature subsets.")
    compare("preprocessing/CV", "Selection/scaling fitted once on all training rows before five-fold CV",
            "Selection/scaling fitted inside each fold's inner training partition",
            "Local CV excludes scoring folds from preprocessing; preserve this improvement in a separate rigorous protocol.")
    compare("class weights", "Inverse-frequency sample weights", "Reference uses inverse-frequency sample weights",
            "This component matches in form; class counts and therefore actual weights differ.")
    compare("XGBoost depth, 34 classes", "46-feature model: 14; selected-feature model: 10", str(config["max_depth"]),
            "The current shared depth-6 configuration is not the author 34-class baseline either.")
    compare("environment", "Paper: Python 3.11.2, XGBoost 3.0.2, sklearn 1.6.1, NumPy 2.1.2, pandas 2.3.0",
            f"Python {experiment['python']}; " + ", ".join(f"{key} {value}" for key, value in experiment["versions"].items()),
            "Exact reproduction should record/version-match the environment; this alone does not explain the gap.")
    small = {label: count for label, count in data["selected_counts"].items() if count < 1000}
    report = {"status": "audit_complete_reproduction_pending", "author_commit": manifest["commit"],
              "paper_url": PAPER_URL, "repository_url": f"{AUTHOR_REPOSITORY}/tree/{manifest['commit']}",
              "version_doi": "10.5281/zenodo.18675263", "concept_doi": "10.5281/zenodo.17070208",
              "local_run": str(run_path.resolve()), "author_settings": settings, "differences": differences,
              "local_raw_labels_below_1000": small,
              "source_warning": "These local raw classes cannot supply the author's per-raw-label minimum; sampling the same local CSV cannot reconstruct the original cohort.",
              "paper_script_discrepancy": "Paper text describes family-level minimum allocation for 8 classes; pinned v3 code samples raw labels before family mapping. Exact released subsets are preferable to an inferred sampler.",
              "hyperparameter_search": "The paper describes grid search; pinned v3 XGBoost ablation functions contain fixed parameter dictionaries, not a reproduced grid-search trace.",
              "test_monitoring": "Author XGBoost fits pass the test split as eval_set but configure no early stopping; this is monitoring and does not itself establish adaptive test-based model selection.",
              "conclusion": "The local 0.7904 reference is not an exact paper reproduction. Multiple material protocol differences are confirmed; their individual score effects have not been isolated.",
              "test_file_read": False, "author_code_executed": False}
    download_path = sources.parent / "download.json"
    if download_path.exists():
        download = _read(download_path)
        report["published_archive_download"] = download
        if download.get("status") == "verified":
            prepared_path = Path(download["path"]).parent / "training_subsets/preparation_report.json"
            if prepared_path.exists():
                prepared = _read(prepared_path)
                report["reference_training_preparation"] = prepared
                if prepared["status"] == "complete":
                    published = prepared["training_subsets"]["8class_train_23features.csv"]["features"]
                    shared = sorted(set(published) & set(training["selected_features"]))
                    report["selected_feature_overlap"] = {
                        "published_count": len(published), "local_count": len(training["selected_features"]),
                        "shared_count": len(shared), "shared_features": shared}
    output_path.mkdir(parents=True, exist_ok=True)
    write_report(output_path / "audit.json", report)
    write_table(output_path / "protocol_comparison.csv", pd.DataFrame(differences))
    write_report(output_path / "author_settings.json", settings)
    lines = ["# Paper baseline audit", "", report["conclusion"], "",
             f"Pinned author commit: `{manifest['commit']}`. Author scripts were parsed, never executed.", "",
             "| Component | Author version 3 | Local eight-class reference | Implication |",
             "|---|---|---|---|"]
    lines += [f"| {r['aspect']} | {r['author_v3']} | {r['local_run']} | {r['interpretation']} |" for r in differences]
    lines += ["", "The local source contains these original labels with fewer than 1,000 rows:", "",
              ", ".join(f"{name}: {count}" for name, count in sorted(small.items())), "",
              report["source_warning"], "", report["paper_script_discrepancy"], "",
              report["hyperparameter_search"], "", report["test_monitoring"], "",
              "Next: verify the versioned subset archive; inspect/extract training files only. A run on those training rows may use an internal validation split without opening the paper test. Exact published-test reproduction remains a separately declared evaluation.", "",
              f"Sources: [paper]({PAPER_URL}), [pinned author source]({report['repository_url']}), "
              "[version-specific archive](https://zenodo.org/records/18675263).", ""]
    if "reference_training_preparation" in report:
        prepared = report["reference_training_preparation"]
        if prepared["status"] == "complete":
            write_report(output_path / "published_training_inventory.json", prepared)
            lines += ["## Verified published training subsets", "",
                      "The archive matches the published size/MD5. Only multiclass training members were extracted and inventoried; no test CSV was parsed.", "",
                      "| Training subset | Rows | Features |", "|---|---:|---:|"]
            for name, entry in prepared["training_subsets"].items():
                lines.append(f"| {name} | {entry['rows']:,} | {entry['feature_count']} |")
            overlap = report["selected_feature_overlap"]
            lines += ["", f"Published and local eight-class selected subsets share {overlap['shared_count']} features "
                      f"out of {overlap['published_count']} published/{overlap['local_count']} local features.", "",
                      "[Training inventory, class counts, and feature names](published_training_inventory.json)", ""]
    (output_path / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report
