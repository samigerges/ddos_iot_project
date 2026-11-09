"""Archive checksum, training-only extraction, and unsafe member rejection."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_iot.attack_families import FAMILY_NAMES, RAW_TO_FAMILY
from ddos_iot.experiments.reference_data import TRAIN_FILES, prepare_reference_training


class ReferenceDataTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, *, link=False):
        path = self.root / "datasets.tar.xz"
        with tarfile.open(path, "w:xz") as archive:
            # Deliberately invalid CSV: parsing any test member would fail.
            test = b"this is not a valid labeled dataset"
            member = tarfile.TarInfo("8class_test_23features.csv"); member.size = len(test)
            archive.addfile(member, io.BytesIO(test))
            for name, classes in TRAIN_FILES.items():
                labels = FAMILY_NAMES if classes == 8 else tuple(RAW_TO_FAMILY)
                frame = pd.DataFrame({f"f{i}": range(len(labels)) for i in range(23)})
                frame["label"] = labels
                data = frame.to_csv(index=False).encode()
                member = tarfile.TarInfo("../" + name)
                if link:
                    member.type = tarfile.SYMTYPE; member.linkname = "outside.csv"
                    archive.addfile(member)
                else:
                    member.size = len(data); archive.addfile(member, io.BytesIO(data))
        record = {"id": 18675263, "files": [{"key": "datasets.tar.xz", "size": path.stat().st_size,
                    "checksum": "md5:" + hashlib.md5(path.read_bytes()).hexdigest(),
                    "links": {"self": "https://example.invalid/archive"}}]}
        record_path = self.root / "record.json"
        record_path.write_text(json.dumps(record), encoding="utf-8")
        return path, record_path

    def test_only_training_members_are_opened_and_paths_stay_inside_output(self):
        archive, record = self.fixture()
        output = self.root / "training"
        result = prepare_reference_training(archive, record, output)
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["test_csv_parsed"])
        self.assertFalse(result["test_members_extracted"])
        self.assertFalse((output / "8class_test_23features.csv").exists())
        self.assertFalse((self.root / "8class_train_23features.csv").exists())
        for name, classes in TRAIN_FILES.items():
            self.assertEqual(result["training_subsets"][name]["rows"], classes)
            self.assertEqual(result["training_subsets"][name]["feature_count"], 23)
            self.assertTrue((output / name).exists())
        with self.assertRaises(FileExistsError):
            prepare_reference_training(archive, record, output)

    def test_corrupt_archive_fails_before_extraction(self):
        archive, record = self.fixture()
        content = bytearray(archive.read_bytes()); content[len(content) // 2] ^= 1
        archive.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "MD5"):
            prepare_reference_training(archive, record, self.root / "output")
        self.assertFalse((self.root / "output").exists())

    def test_training_symlink_is_rejected(self):
        archive, record = self.fixture(link=True)
        with self.assertRaisesRegex(ValueError, "non-regular"):
            prepare_reference_training(archive, record, self.root / "output")
        report = json.loads((self.root / "output/preparation_report.json").read_text())
        self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
