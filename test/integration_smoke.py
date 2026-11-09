"""Real Julia runs through both wrappers; requires Julia dependencies and Rscript.

Run with ``python3 test/integration_smoke.py``. Set JULIA to a Julia executable
when it is not on PATH. JULIA_DEPOT_PATH is inherited without modification.
These tiny runs validate the interface and reproducibility, not MCMC mixing.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "interfaces" / "python"))
from mixture_workbench import read_output  # noqa: E402

JULIA = os.environ.get("JULIA") or shutil.which("julia")
RSCRIPT = os.environ.get("RSCRIPT") or shutil.which("Rscript")
TIMEOUT = int(os.environ.get("MIXTURE_SMOKE_TIMEOUT", "600"))


def invoke(language: str, config: Path, data: Path, output: Path) -> subprocess.CompletedProcess:
    if language == "python":
        command = [sys.executable, str(ROOT / "interfaces/python/mixture_workbench.py")]
    else:
        command = [RSCRIPT, str(ROOT / "interfaces/R/run_sampler.R")]
    command += ["--config", str(config), "--data", str(data), "--output", str(output),
                "--julia", JULIA]
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT)


def check_fit(output: Path, method: str, n: int):
    fit = read_output(output)
    assert fit.metadata["model"] == method, fit.metadata
    assert len(fit.metadata["source_digest"]) == 64
    assert int(fit.metadata["n"]) == n
    assert int(fit.metadata["chains"]) == 1
    assert len(fit.draws) == 4, fit.draws
    assert [row["iteration"] for row in fit.draws] == [6, 8, 10, 12]
    for row in fit.draws:
        assert row["chain"] == 1
        assert 1 <= row["k"] <= n and row["M"] >= row["k"], row
        assert -1e-12 <= row["entropy"] <= math.log(row["k"]) + 1e-10, row
        if method == "strauss_general":
            assert math.isnan(row["log_target"]), row
            assert fit.metadata["log_target_definition"] == "unavailable_NaN"
        else:
            assert math.isfinite(row["log_target"]), row
            assert fit.metadata["log_target_definition"] == "cold_collapsed_target"
    assert fit.allocations is not None and len(fit.allocations) == 4 * n
    for draw in fit.draws:
        allocations = [row for row in fit.allocations if row["iteration"] == draw["iteration"]]
        assert {row["observation"] for row in allocations} == set(range(1, n + 1))
        assert {row["cluster"] for row in allocations} == set(range(1, draw["k"] + 1))
    split_merge = next(row for row in fit.diagnostics if row["metric"] == "split_merge")
    assert 0 <= split_merge["accepted"] <= split_merge["attempted"]
    if method in ("mfm", "dpp_marginal_sm"):
        assert split_merge["attempted"] > 0, split_merge
    else:
        assert split_merge["attempted"] == 0 and split_merge["status"] == "unassessed"
    if method == "mfm":
        with (output / "transport.csv").open(newline="") as handle:
            transport = list(csv.DictReader(handle))
        assert len(transport) == 8
        assert {float(row["beta"]) for row in transport} == {1.0, 0.8}
    return fit


def main() -> None:
    if not JULIA or not RSCRIPT:
        raise SystemExit("Real smoke tests require Julia and Rscript; set JULIA/RSCRIPT or PATH")
    with tempfile.TemporaryDirectory(prefix="mixture real smoke with spaces ") as temporary:
        directory = Path(temporary).resolve()
        data = directory / "Gaussian data's.csv"
        shutil.copyfile(ROOT / "examples/gaussian/data.csv", data)
        with data.open(newline="") as handle:
            n = sum(1 for _ in csv.DictReader(handle))
        common = (
            "iterations = 12\nwarmup = 4\nthin = 2\nchains = 1\nseed = 42\n"
            "initial_clusters = 2\nsave_allocations = true\n"
        )
        for method in ("mfm", "dpp_marginal_sm", "strauss_general"):
            config = directory / f"{method} settings.toml"
            extra = "split_merge = true\nbetas = [1.0, 0.8]\nswap_every = 2\n" if method == "mfm" else ""
            config.write_text(f'method = "{method}"\n' + common + extra)
            outputs = []
            for language in ("python", "R"):
                print(f"Real smoke: {language} -> Julia {method}", flush=True)
                output = directory / f"{method} {language}'s output"
                process = invoke(language, config, data, output)
                assert process.returncode == 0, process.stdout + process.stderr
                check_fit(output, method, n)
                outputs.append(output)
            # Identical inputs and seeds must give identical draws and allocations
            # through either language wrapper. Runtime metadata is intentionally excluded.
            for filename in ("draws.csv", "allocations.csv", "diagnostics.csv"):
                assert (outputs[0] / filename).read_bytes() == (outputs[1] / filename).read_bytes(), (
                    method, filename, "R/Python runs disagree for the same seed"
                )

        invalid = (
            ("python", 'method = "mfm"\nzero_inflation = true\n', "unknown settings"),
            ("R", 'method = "hdp"\n', "unsupported method"),
            ("python", 'method = "strauss_general"\nsplit_merge = true\n', "supported only"),
            ("R", 'method = "dpp_marginal_sm"\nbetas = [1.0, 0.8]\n', "implemented here only"),
        )
        for index, (language, settings, expected) in enumerate(invalid, 1):
            print(f"Real smoke: reject unsupported configuration {index}", flush=True)
            config = directory / f"invalid settings {index}.toml"
            config.write_text(settings + common)
            output = directory / f"invalid output {index}"
            process = invoke(language, config, data, output)
            assert process.returncode != 0, process.stdout + process.stderr
            assert expected in process.stderr, process.stderr
            assert not output.exists(), "Validation failure created an output directory"
    print("Real Julia/Python/R smoke tests passed", flush=True)


if __name__ == "__main__":
    main()
