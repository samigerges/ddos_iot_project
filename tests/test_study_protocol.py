"""Cross-task final freeze prevents information from one test selecting the other task."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddos_iot.experiments.storage import object_digest, write_report
from ddos_iot.experiments.study_protocol import freeze_research_protocol, evaluate_research_protocol, assess_recall_policy, select_under_recall_policy


class GlobalProtocolTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="protocol_test_", dir=ROOT / "tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.choices = {}
        for classes in (8, 34):
            source = self.root / f"study{classes}"
            confirmation = source / "confirmation"
            self.choices[classes] = confirmation
            write_report(source / "config.json", {"num_classes": classes})
            write_report(source / "report.json", {"status": "complete"})
            write_report(source / "seed_42/selection.json", {"candidate": "paper_fixed"})
            selection = {"candidate": "paper_fixed", "seeds": [42, 123, 2024], "test_file_read": False}
            write_report(confirmation / "selection.json", selection)
            write_report(confirmation / "report.json", {"status": "complete", "source_study": str(source),
                "source_seed": 42, "selection_sha256": object_digest(selection),
                "results": [{"candidate": "paper_fixed", "seed": seed, "macro_f1": .89} for seed in selection["seeds"]]})
            reference = self.root / f"outputs/studies/paper_fidelity_v3/{classes}class"
            reference.mkdir(parents=True)
            (reference / "model.joblib").write_bytes(b"test fixture only")
        for name in ("archive", "record", "development"):
            write_report(self.root / name, {"status": "complete"})
        module = self.root / "src/ddos_iot/multiclass.py"
        module.parent.mkdir(parents=True)
        module.write_text("# global-freeze fixture\n")
        self.addCleanup(patch.stopall)
        patch("ddos_iot.experiments.study_protocol.PROJECT_ROOT", self.root).start()
        patch("ddos_iot.experiments.study_protocol.paper_reference_descriptor",
              side_effect=lambda p: {"directory": str(p), "num_classes": int(p.name.removesuffix("class"))}).start()

    def freeze(self):
        return freeze_research_protocol(self.choices, self.root / "final", archive=self.root / "archive",
            record=self.root / "record", recall_policy="preserve all reference recalls/FPR in every seed",
            paper_python=self.root / "python.exe", development_report=self.root / "development")

    def test_changed_second_task_blocks_first_test(self):
        self.freeze()
        write_report(self.choices[34] / "selection.json", {"candidate": "changed"})
        with patch("ddos_iot.experiments.study_protocol.evaluate_study_test") as evaluate:
            with self.assertRaisesRegex(ValueError, "frozen input changed"):
                evaluate_research_protocol(self.root / "final")
            evaluate.assert_not_called()

    def test_library_environment_change_blocks_first_test(self):
        self.freeze()
        with patch("ddos_iot.experiments.study_protocol._environment", return_value={"python": "changed"}), \
             patch("ddos_iot.experiments.study_protocol.evaluate_study_test") as evaluate:
            with self.assertRaisesRegex(ValueError, "environment changed"):
                evaluate_research_protocol(self.root / "final")
            evaluate.assert_not_called()

    def test_comparison_only_evidence_cannot_be_silently_promoted(self):
        path = self.choices[34] / "selection.json"
        selection = json.loads(path.read_text())
        selection["comparison_only"] = True
        write_report(path, selection)
        report_path = path.with_name("report.json")
        report = json.loads(report_path.read_text())
        report["selection_sha256"] = object_digest(selection)
        write_report(report_path, report)
        with self.assertRaisesRegex(ValueError, "explicit policy selection"):
            self.freeze()
        self.assertFalse((self.root / "final").exists())

    def test_existing_second_test_blocks_global_freeze(self):
        (self.root / "data/raw/paper_v3/final_tests/34class").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.freeze()
        self.assertFalse((self.root / "final").exists())

    def test_execution_keeps_both_frozen_choices_and_reuses_completed_stages(self):
        protocol = self.freeze()

        def evaluate(confirmation, output, **kwargs):
            self.assertEqual(kwargs["latency_repeats"], 1)
            self.assertTrue((self.root / "final/protocol.json").exists())
            write_report(output / "report.json", {"status": "complete"})
            write_report(output / "protocol.json", {"frozen": True})

        def score(command, **kwargs):
            write_report(Path(command[command.index("--output-dir") + 1]) / "report.json", {"status": "complete"})

        def uncertainty(reference, candidate, output):
            self.assertEqual(reference, candidate)  # These fixture choices retain the baseline.
            write_report(output / "report.json", {"status": "complete"})

        with patch("ddos_iot.experiments.study_protocol.evaluate_study_test", side_effect=evaluate) as final, \
             patch("ddos_iot.experiments.study_protocol.subprocess.run", side_effect=score) as paper, \
             patch("ddos_iot.experiments.study_protocol.report_test_uncertainty", side_effect=uncertainty) as bootstrap:
            result = evaluate_research_protocol(self.root / "final")
            self.assertEqual(result["status"], "complete")
            self.assertEqual([call.args[0] for call in final.call_args_list], [self.choices[8], self.choices[34]])
            evaluate_research_protocol(self.root / "final")
            self.assertEqual((final.call_count, paper.call_count, bootstrap.call_count), (2, 2, 2))
        self.assertEqual(protocol["uncertainty"]["model_seed"], 42)

    def test_policy_selection_preserves_original_guard_failures_and_frozen_settings(self):
        base_folder = self.choices[34]
        source = base_folder.parent
        parent = {"protected_labels": ["minority"]}
        write_report(source / "seed_42/selection.json", parent)
        candidates, references = RecallPolicyTests().fixture()
        base = json.loads((base_folder / "report.json").read_text())
        base.update(results=[{**r, "candidate": "paper_fixed"} for r in references],
                    source_selection_sha256=object_digest(parent))
        write_report(base_folder / "report.json", base)
        advanced = source / "specialist"
        frozen = {"candidates": {"specialist_075": {"weight_power": .75, "rounds": 500, "depth": 10}}}
        write_report(advanced / "selection.json", frozen)
        evidence = source / "comparison"
        previous = {"comparison_only": True}
        write_report(evidence / "selection.json", previous)
        report = {"status": "complete", "selection_sha256": object_digest(previous),
            "source_study": str(source), "base_confirmation": str(base_folder), "advanced_source": str(advanced),
            "source_selection_sha256": object_digest(parent), "advanced_source_selection_sha256": object_digest(frozen),
            "results": [{**r, "candidate": "specialist_075", "reference_constraints_pass": False} for r in candidates]}
        write_report(evidence / "report.json", report)
        output = source / "policy_confirmation"
        chosen = select_under_recall_policy(evidence, "specialist_075", output,
            maximum_recall_loss=.01, scope="mean", policy_note="Fixture policy established before test.")
        saved = json.loads((output / "report.json").read_text())
        self.assertTrue(chosen["advanced_confirmed"])
        self.assertEqual(chosen["advanced_source_selection"], frozen)
        self.assertTrue(all(not row["reference_constraints_pass"] for row in saved["results"]))
        self.assertEqual(json.loads((evidence / "selection.json").read_text()), previous)


class RecallPolicyTests(unittest.TestCase):
    def fixture(self):
        references = [{"seed": seed, "macro_f1": .89, "benign_false_positive_rate": .04,
            "recall_by_class": {"minority": .6}} for seed in (42, 123, 2024)]
        candidates = [{**r, "macro_f1": .9, "recall_by_class": {"minority": .6 + delta}}
                      for r, delta in zip(references, (-.0125, .01, .01))]
        return candidates, references

    def test_mean_and_every_seed_policies_are_distinct(self):
        candidates, references = self.fixture()
        strict = assess_recall_policy(candidates, references, ["minority"])
        mean = assess_recall_policy(candidates, references, ["minority"], maximum_recall_loss=.01, scope="mean")
        self.assertFalse(strict["accepted"])
        self.assertTrue(mean["accepted"])
        self.assertAlmostEqual(mean["mean_recall_deltas"]["minority"], .0025)
        self.assertAlmostEqual(mean["worst_seed_recall_deltas"]["minority"], -.0125)

    def test_false_alarm_increase_in_one_seed_still_rejects(self):
        candidates, references = self.fixture()
        candidates[1]["benign_false_positive_rate"] = .041
        self.assertFalse(assess_recall_policy(candidates, references, ["minority"], maximum_recall_loss=.01, scope="mean")["accepted"])

    def test_missing_or_duplicate_seed_does_not_become_a_mean(self):
        candidates, references = self.fixture()
        candidates[1]["seed"] = 42
        with self.assertRaisesRegex(ValueError, "three paired seeds"):
            assess_recall_policy(candidates, references, ["minority"], maximum_recall_loss=.01, scope="mean")


if __name__ == "__main__":
    unittest.main()
