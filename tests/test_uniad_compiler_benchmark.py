"""Focused tests for benchmark provenance reporting."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).parents[1]
BENCHMARK_PATH = ROOT / "tools" / "uniad_compiler" / "benchmark_ffn.py"
SPEC = importlib.util.spec_from_file_location("uniad_benchmark_ffn", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class CompilerSourceProvenanceTests(unittest.TestCase):
    def test_explicit_checkout_reports_mismatched_head_and_hashes_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src" / "gac" / "runtime.py"
            source.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n", encoding="utf-8")
            expected_source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            actual_revision = "1" * 40
            with patch.object(
                benchmark,
                "_git_state",
                return_value=(actual_revision, False),
            ):
                report = benchmark._compiler_source_provenance(
                    root, source.parent / "__init__.py"
                )

        gac_source = report["gac_source"]
        self.assertEqual(report["expected_gac_commit"], benchmark.PINNED_GAC_COMMIT)
        self.assertEqual(gac_source["mode"], "checkout")
        self.assertEqual(gac_source["actual_revision"], actual_revision)
        self.assertFalse(gac_source["revision_matches_expected_commit"])
        self.assertFalse(gac_source["tracked_files_dirty"])
        self.assertTrue(gac_source["python_source_hashes_collected"])
        self.assertEqual(gac_source["python_source_hash_count"], 1)
        self.assertEqual(
            report["source_sha256"]["gac/src/gac/runtime.py"],
            expected_source_hash,
        )

    def test_installed_package_does_not_claim_gac_source_hashes(self):
        installed = {
            "distribution_name": "gac-heterogeneous",
            "distribution_version": "0.1.0",
            "pep610_vcs_commit": None,
        }
        with patch.object(
            benchmark, "_installed_gac_distribution", return_value=installed
        ):
            report = benchmark._compiler_source_provenance(
                None, Path("site-packages/gac/__init__.py")
            )

        gac_source = report["gac_source"]
        self.assertEqual(gac_source["mode"], "installed-package")
        self.assertIsNone(gac_source["actual_revision"])
        self.assertIsNone(gac_source["revision_matches_expected_commit"])
        self.assertFalse(gac_source["python_source_hashes_collected"])
        self.assertEqual(gac_source["python_source_hash_count"], 0)
        self.assertFalse(
            any(name.startswith("gac/") for name in report["source_sha256"])
        )


if __name__ == "__main__":
    unittest.main()
