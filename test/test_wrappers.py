"""Wrapper contract tests; no Julia dependencies or numerical model assumptions."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "interfaces" / "python" / "mixture_workbench.py"
spec = importlib.util.spec_from_file_location("mixture_workbench", WRAPPER)
workbench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = workbench
spec.loader.exec_module(workbench)

FAKE_JULIA = r'''#!/usr/bin/env python3
import csv
import json
from pathlib import Path
import sys

args = sys.argv[1:]
assert args[0] == "--startup-file=no"
assert args[1].startswith("--project=")
assert Path(args[2]).is_file()
assert args[3::2] == ["--config", "--data", "--output"]
config, data, output = map(Path, args[4::2])
assert config.is_file() and data.is_file()
if config.read_text().strip() == "fail = true":
    print("deliberate fake engine failure", file=sys.stderr)
    sys.exit(17)
output.mkdir(parents=True)
(output / "argv.json").write_text(json.dumps(args))
with (output / "draws.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["chain", "iteration", "k", "M", "entropy", "log_target"])
    writer.writerow([1, 10, 2, 3, 0.693, -12.5])
with (output / "diagnostics.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["chain", "metric", "attempted", "accepted", "value", "status"])
    writer.writerow([1, "split_merge", 4, 1, 0.25, "assessed"])
with (output / "metadata.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["key", "value"])
    writer.writerow(["schema_version", "1"])
    writer.writerow(["run_status", "completed"])
    writer.writerow(["description", 'quoted, value with "text"'])
(output / "metadata.toml").write_text("schema_version = 1\n")
if config.read_text().strip() == "allocations = true":
    (output / "allocations.csv").write_text(
        "chain,iteration,observation,cluster\n1,10,1,2\n"
    )
'''


@unittest.skipIf(os.name == "nt", "Fake executable uses a POSIX shebang")
class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="workbench wrappers ")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.project = self.directory / "project with spaces and ' quote"
        (self.project / "bin").mkdir(parents=True)
        (self.project / "bin" / "run.jl").write_text("# test placeholder\n")
        self.config = self.directory / "settings with $literal.toml"
        self.config.write_text("seed = 123\n")
        self.data = self.directory / "data with ' quote.csv"
        self.data.write_text("x1\n1\n2\n")
        self.output = self.directory / "output with spaces"
        self.fake_julia = self.directory / "fake julia"
        self.fake_julia.write_text(FAKE_JULIA)
        self.fake_julia.chmod(0o755)

    def run_fit(self):
        return workbench.run_sampler(
            self.config, self.data, self.output,
            julia=self.fake_julia, project_root=self.project,
        )

    def test_argv_paths_and_results(self):
        fit = self.run_fit()
        arguments = json.loads((self.output / "argv.json").read_text())
        self.assertEqual(arguments, [
            "--startup-file=no", f"--project={self.project}",
            str(self.project / "bin" / "run.jl"),
            "--config", str(self.config), "--data", str(self.data),
            "--output", str(self.output),
        ])
        self.assertEqual(fit.draws[0]["k"], 2)
        self.assertIsInstance(fit.draws[0]["k"], int)
        self.assertEqual(fit.draws[0]["log_target"], -12.5)
        self.assertEqual(fit.diagnostics[0]["accepted"], 1)
        self.assertEqual(fit.diagnostics[0]["status"], "assessed")
        self.assertEqual(fit.metadata["schema_version"], "1")
        self.assertEqual(fit.metadata["description"], 'quoted, value with "text"')
        self.assertIsNone(fit.allocations)

    def test_optional_allocations_and_reload(self):
        self.config.write_text("allocations = true\n")
        self.run_fit()
        fit = workbench.read_output(self.output)
        self.assertEqual(fit.allocations[0], {
            "chain": 1, "iteration": 10, "observation": 1, "cluster": 2,
        })

    def test_alternate_entrypoint(self):
        entrypoint = Path("examples") / "a model with spaces" / "sample.jl"
        (self.project / entrypoint).parent.mkdir(parents=True)
        (self.project / entrypoint).write_text("# alternate implemented model\n")
        workbench.run_sampler(
            self.config, self.data, self.output, julia=self.fake_julia,
            project_root=self.project, entrypoint=entrypoint,
        )
        arguments = json.loads((self.output / "argv.json").read_text())
        self.assertEqual(arguments[2], str(self.project / entrypoint))

    def test_missing_entrypoint_fails_before_execution(self):
        with self.assertRaises(FileNotFoundError):
            workbench.run_sampler(
                self.config, self.data, self.output, julia=self.fake_julia,
                project_root=self.project, entrypoint="missing.jl",
            )
        self.assertFalse(self.output.exists())

    def test_nonzero_exit_propagates(self):
        self.config.write_text("fail = true\n")
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            self.run_fit()
        self.assertEqual(caught.exception.returncode, 17)
        self.assertFalse(self.output.exists())

    def test_cli_forwards_nonzero_exit(self):
        self.config.write_text("fail = true\n")
        process = subprocess.run([
            sys.executable, str(WRAPPER), "--config", str(self.config),
            "--data", str(self.data), "--output", str(self.output),
            "--julia", str(self.fake_julia), "--project-root", str(self.project),
        ], capture_output=True, text=True)
        self.assertEqual(process.returncode, 17)
        self.assertIn("deliberate fake engine failure", process.stderr)

    def test_cli_success(self):
        process = subprocess.run([
            sys.executable, str(WRAPPER), "--config", str(self.config),
            "--data", str(self.data), "--output", str(self.output),
            "--julia", str(self.fake_julia), "--project-root", str(self.project),
            "--entrypoint", "bin/run.jl",
        ], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("Read 1 draws", process.stdout)

    def test_missing_inputs_fail_before_execution(self):
        self.config.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_fit()
        self.assertFalse(self.output.exists())

    def test_missing_draw_columns_are_rejected(self):
        self.run_fit()
        (self.output / "draws.csv").write_text("chain,k\n1,2\n")
        with self.assertRaisesRegex(ValueError, "Missing columns"):
            workbench.read_output(self.output)

    def test_unfinished_and_failed_outputs_are_rejected(self):
        self.run_fit()
        for status in ("running", "failed"):
            with self.subTest(status=status):
                (self.output / "metadata.csv").write_text(
                    f"key,value\nschema_version,1\nrun_status,{status}\n"
                )
                with self.assertRaisesRegex(ValueError, "Run is not completed"):
                    workbench.read_output(self.output)

    def test_unsupported_output_schema_is_rejected(self):
        self.run_fit()
        (self.output / "metadata.csv").write_text(
            "key,value\nschema_version,2\nrun_status,completed\n"
        )
        with self.assertRaisesRegex(ValueError, "Unsupported output schema_version"):
            workbench.read_output(self.output)

    def test_non_numeric_draws_are_rejected(self):
        self.run_fit()
        with (self.output / "draws.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["chain", "iteration", "k", "M", "entropy", "log_target"])
            writer.writerow([1, 10, "not a number", 3, 0.693, -12.5])
        with self.assertRaisesRegex(ValueError, "Invalid numeric value"):
            workbench.read_output(self.output)


if __name__ == "__main__":
    unittest.main()
