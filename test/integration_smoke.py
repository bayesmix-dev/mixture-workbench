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


def invoke(
    language: str, config: Path, data: Path, output: Path, entrypoint: str | None = None
) -> subprocess.CompletedProcess:
    if language == "python":
        command = [sys.executable, str(ROOT / "interfaces/python/mixture_workbench.py")]
    else:
        command = [RSCRIPT, str(ROOT / "interfaces/R/run_sampler.R")]
    command += ["--config", str(config), "--data", str(data), "--output", str(output),
                "--julia", JULIA]
    if entrypoint is not None:
        command += ["--entrypoint", entrypoint]
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


def check_poisson_fit(output: Path, counts: list[int], components: int):
    fit = read_output(output)
    assert fit.metadata["schema_version"] == "1"
    assert fit.metadata["run_status"] == "completed"
    assert fit.metadata["model"] == "poisson_lognormal_fixed_finite"
    assert int(fit.metadata["components"]) == components
    assert int(fit.metadata["n"]) == len(counts) and fit.metadata["d"] == "1"
    assert fit.metadata["empirical_bayes"] == "false"
    assert "matching_eta_columns" in fit.metadata["allocation_labels"]
    for key in ("source_digest", "extension_source_digest", "input_sha256", "config_sha256"):
        assert len(fit.metadata[key]) == 64, key
    assert len(fit.draws) == 4
    assert [row["iteration"] for row in fit.draws] == [6, 8, 10, 12]
    expected_columns = {"chain", "iteration", "k", "M", "entropy", "log_target"}
    expected_columns.update(f"eta_{h}" for h in range(1, components + 1))
    assert all(set(draw) == expected_columns for draw in fit.draws)
    assert fit.allocations is not None and len(fit.allocations) == 4 * len(counts)

    gamma = float(fit.metadata["weight_shape"])
    mu = float(fit.metadata["base_mean"])
    sd = float(fit.metadata["base_sd"])
    for draw in fit.draws:
        assert draw["chain"] == 1 and draw["M"] == components
        assert 1 <= draw["k"] <= components
        eta = [draw[f"eta_{h}"] for h in range(1, components + 1)]
        assert all(isinstance(value, float) and math.isfinite(value) for value in eta)
        allocation = sorted(
            (row for row in fit.allocations if row["iteration"] == draw["iteration"]),
            key=lambda row: row["observation"],
        )
        assert [row["observation"] for row in allocation] == list(range(1, len(counts) + 1))
        assert all(row["chain"] == 1 for row in allocation)
        labels = [row["cluster"] for row in allocation]
        assert list(dict.fromkeys(labels)) == list(range(1, draw["k"] + 1))
        occupancies = [labels.count(h) for h in range(1, components + 1)]
        assert all(occupancies[h] == 0 for h in range(draw["k"], components))

        # Reconstruct the labelled joint target independently from the exported
        # allocations and eta columns. This checks their correspondence, including
        # the base density for empty components, rather than only checking shape.
        target = math.lgamma(components * gamma) - math.lgamma(len(counts) + components * gamma)
        target += sum(math.lgamma(nh + gamma) - math.lgamma(gamma) for nh in occupancies)
        target += sum(-math.log(sd) - 0.5 * math.log(2 * math.pi)
                      - 0.5 * ((value - mu) / sd) ** 2 for value in eta)
        target += sum(y * eta[label - 1] - math.exp(eta[label - 1]) - math.lgamma(y + 1)
                      for y, label in zip(counts, labels))
        assert math.isclose(draw["log_target"], target, rel_tol=1e-11, abs_tol=1e-9), (
            draw, target, "Exported allocations do not match the eta columns"
        )
        entropy = -sum(nh / len(counts) * math.log(nh / len(counts))
                       for nh in occupancies if nh)
        assert math.isclose(draw["entropy"], entropy, abs_tol=1e-12)

    rate = next(row for row in fit.diagnostics if row["metric"] == "log_rate_mh")
    refresh = next(row for row in fit.diagnostics if row["metric"] == "empty_component_prior_refresh")
    mixing = next(row for row in fit.diagnostics if row["metric"] == "mixing")
    assert 0 <= rate["accepted"] <= rate["attempted"]
    assert rate["attempted"] + refresh["attempted"] == 12 * components
    assert mixing["status"] == "unassessed"
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

        entrypoint = "examples/poisson_lognormal/run.jl"
        count_data = directory / "Poisson count data's.csv"
        shutil.copyfile(ROOT / "examples/poisson_lognormal/counts.csv", count_data)
        with count_data.open(newline="") as handle:
            counts = [int(row["count"]) for row in csv.DictReader(handle)]
        poisson_common = (
            "weight_shape = 0.5\nbase_mean = 1.0\nbase_sd = 1.5\nproposal_sd = 0.6\n"
            "iterations = 12\nwarmup = 4\nthin = 2\nchains = 1\nseed = 42\n"
            "save_allocations = true\n"
        )
        config = directory / "Poisson extension settings.toml"
        config.write_text('method = "poisson_lognormal"\ncomponents = 3\n' + poisson_common)
        outputs = []
        for language in ("python", "R"):
            print(f"Real smoke: {language} -> Julia Poisson extension", flush=True)
            output = directory / f"Poisson {language}'s output"
            process = invoke(language, config, count_data, output, entrypoint=entrypoint)
            assert process.returncode == 0, process.stdout + process.stderr
            check_poisson_fit(output, counts, components=3)
            outputs.append(output)
        for filename in ("draws.csv", "allocations.csv", "diagnostics.csv"):
            assert (outputs[0] / filename).read_bytes() == (outputs[1] / filename).read_bytes(), (
                filename, "R/Python extension runs disagree for the same seed"
            )

        invalid_poisson = (
            ("python", 'method = "poisson_lognormal"\ncomponents = 3\nzero_inflation = true\n',
             "unsupported Poisson-lognormal settings"),
            ("R", 'method = "hdp"\ncomponents = 3\n', "method must be 'poisson_lognormal'"),
            ("python", 'method = "poisson_lognormal"\ncomponents = 0\n',
             "components must be an integer >= 1"),
            ("R", 'method = "poisson_lognormal"\ncomponents = 3\nsplit_merge = true\n',
             "unsupported Poisson-lognormal settings"),
        )
        for index, (language, settings, expected) in enumerate(invalid_poisson, 1):
            print(f"Real smoke: reject unsupported Poisson configuration {index}", flush=True)
            config = directory / f"invalid Poisson settings {index}.toml"
            config.write_text(settings + poisson_common)
            output = directory / f"invalid Poisson output {index}"
            process = invoke(language, config, count_data, output, entrypoint=entrypoint)
            assert process.returncode != 0, process.stdout + process.stderr
            assert expected in process.stderr, process.stderr
            assert not output.exists(), "Poisson validation failure created an output directory"
    print("Real Julia/Python/R smoke tests passed", flush=True)


if __name__ == "__main__":
    main()
