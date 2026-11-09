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



def check_partition_fit(output: Path, observations: list[float], model: str):
    fit = read_output(output)
    assert fit.metadata["model"] == model
    assert fit.metadata["run_status"] == "completed"
    assert fit.metadata["total_components"] == "infinite"
    assert fit.metadata["empirical_bayes"] == "false"
    assert fit.metadata["mixing_assessment"] == "unassessed"
    assert fit.metadata["log_target_definition"] in ("partition_eppf_times_integrated_gaussian_likelihood",
                                                      "partition_eppf_times_occupied_base_and_likelihood")
    assert len(fit.draws) == 4
    assert [r["iteration"] for r in fit.draws] == [6, 8, 10, 12]
    assert fit.allocations is not None and len(fit.allocations) == 4 * len(observations)
    alpha = float(fit.metadata["strength"])
    discount = float(fit.metadata["discount"])
    variance = float(fit.metadata["observation_sd"]) ** 2
    base_mean = float(fit.metadata["base_mean"])
    base_variance = float(fit.metadata["base_sd"]) ** 2
    for draw in fit.draws:
        assert draw["chain"] == 1
        rows = sorted((r for r in fit.allocations if r["iteration"] == draw["iteration"]),
                      key=lambda r: r["observation"])
        labels = [r["cluster"] for r in rows]
        assert [r["observation"] for r in rows] == list(range(1, len(observations) + 1))
        assert list(dict.fromkeys(labels)) == list(range(1, draw["k"] + 1))
        assert draw["M"] >= draw["k"]
        if fit.metadata["M_definition"] == "represented_occupied_atoms_equals_k":
            assert draw["M"] == draw["k"]
        # Independent sequential prediction: CRP/PY seating probabilities and
        # sequential Normal predictive densities reproduce the partition target.
        counts, means, variances = {}, {}, {}
        target = 0.0
        for i, (label, y) in enumerate(zip(labels, observations)):
            if i:
                seating = counts[label] - discount if label in counts else alpha + discount * len(counts)
                target += math.log(seating) - math.log(alpha + i)
            mean = means.get(label, base_mean)
            v = variances.get(label, base_variance)
            target += -0.5 * (math.log(2 * math.pi * (variance + v)) + (y - mean) ** 2 / (variance + v))
            posterior_variance = 1 / (1 / v + 1 / variance)
            means[label] = posterior_variance * (mean / v + y / variance)
            variances[label] = posterior_variance
            counts[label] = counts.get(label, 0) + 1
        assert math.isclose(draw.get("partition_log_target", draw["log_target"]), target,
                            rel_tol=1e-11, abs_tol=1e-9), (draw, target)
        entropy = -sum(n / len(labels) * math.log(n / len(labels)) for n in counts.values())
        assert math.isclose(draw["entropy"], entropy, abs_tol=1e-12)
    return fit



def check_neal_fit(output: Path, observations: list[float], likelihood: str, method: str):
    model = f"dirichlet_process_{likelihood}"
    fit = (check_partition_fit(output, observations, model) if likelihood == "gaussian_location"
           else read_output(output))
    assert fit.metadata["model"] == model and fit.metadata["sampler"] == method
    assert len(fit.draws) == 4 and fit.allocations is not None
    if method == "neal3":
        assert fit.parameters is None
        return fit
    assert fit.parameters is not None
    assert fit.metadata["log_target_definition"] == "partition_eppf_times_occupied_base_and_likelihood"
    alpha = float(fit.metadata["strength"])
    mu = float(fit.metadata["base_mean"])
    sd = float(fit.metadata["base_sd"])
    for draw in fit.draws:
        assert draw["M"] == draw["k"]
        atoms = [r for r in fit.parameters if r["iteration"] == draw["iteration"]]
        assert len(atoms) == draw["k"]
        theta = {r["cluster"]: r["theta"] for r in atoms}
        assert set(theta) == set(range(1, draw["k"] + 1))
        rows = sorted((r for r in fit.allocations if r["iteration"] == draw["iteration"]),
                      key=lambda r: r["observation"])
        counts = {}
        target = 0.0
        for i, (y, row) in enumerate(zip(observations, rows)):
            label = row["cluster"]
            if i:
                target += math.log(counts.get(label, alpha)) - math.log(alpha + i)
            counts[label] = counts.get(label, 0) + 1
            value = theta[label]
            if likelihood == "gaussian_location":
                sigma = float(fit.metadata["observation_sd"])
                target += -math.log(sigma) - 0.5 * math.log(2 * math.pi) - 0.5 * ((y - value) / sigma) ** 2
            else:
                target += y * value - math.exp(value) - math.lgamma(y + 1)
        target += sum(-math.log(sd) - 0.5 * math.log(2 * math.pi) - 0.5 * ((t - mu) / sd) ** 2
                      for t in theta.values())
        assert math.isclose(draw["log_target"], target, rel_tol=1e-11, abs_tol=1e-9), (draw, target)
    return fit



def check_slice_fit(output: Path, observations: list[float]):
    fit = check_partition_fit(output, observations, "dirichlet_process_gaussian_location")
    assert fit.metadata["sampler"] == "slice_efficient"
    assert fit.metadata["log_target_is_projection"] == "true"
    assert fit.metadata["M_definition"] == "materialized_ordered_gem_prefix_including_empty_atoms"
    assert fit.metadata["tail_stopping"] == "residual_mass_below_minimum_slice_no_fixed_truncation"
    assert fit.parameters is not None
    alpha = float(fit.metadata["strength"])
    mu, sd = float(fit.metadata["base_mean"]), float(fit.metadata["base_sd"])
    sigma = float(fit.metadata["observation_sd"])
    for draw in fit.draws:
        atoms = [r for r in fit.parameters if r["iteration"] == draw["iteration"]]
        assert len(atoms) == draw["k"]
        by_label = {r["cluster"]: r for r in atoms}
        assert set(by_label) == set(range(1, draw["k"] + 1))
        gems = [r["gem_index"] for r in atoms]
        assert len(set(gems)) == draw["k"] and max(gems) <= draw["M"]
        rows = sorted((r for r in fit.allocations if r["iteration"] == draw["iteration"]),
                      key=lambda r: r["observation"])
        counts = [0] * max(gems)
        target = 0.0
        for y, row in zip(observations, rows):
            atom = by_label[row["cluster"]]
            counts[atom["gem_index"] - 1] += 1
            target += -math.log(sigma) - 0.5 * math.log(2 * math.pi) - 0.5 * ((y - atom["theta"]) / sigma) ** 2
        tail = len(observations)
        for count in counts:
            tail -= count
            a, b = 1 + count, alpha + tail
            target += math.log(alpha) + math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        target += sum(-math.log(sd) - 0.5 * math.log(2 * math.pi) - 0.5 * ((r["theta"] - mu) / sd) ** 2
                      for r in atoms)
        assert math.isclose(draw["ordered_log_target"], target, rel_tol=1e-11, abs_tol=1e-9), (draw, target)
    sticks = next(r for r in fit.diagnostics if r["metric"] == "slice_stick_draws")
    assert sticks["attempted"] >= 12 and sticks["accepted"] == sticks["attempted"]
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
        entrypoint = "examples/pitman_yor/run.jl"
        observations = [-1.2, -0.4, 0.0, 0.9, 1.3]
        species_data = directory / "Species data's.csv"
        species_data.write_text("y\n" + "\n".join(map(str, observations)) + "\n")
        settings = ('method = "pitman_yor"\nstrength = -0.2\ndiscount = 0.4\n'
                    'observation_sd = 0.7\nbase_mean = 0.0\nbase_sd = 2.0\n' + common)
        config = directory / "Pitman-Yor settings.toml"
        config.write_text(settings)
        outputs = []
        for language in ("python", "R"):
            print(f"Real smoke: {language} -> Julia Pitman-Yor", flush=True)
            output = directory / f"Pitman-Yor {language}'s output"
            process = invoke(language, config, species_data, output, entrypoint=entrypoint)
            assert process.returncode == 0, process.stdout + process.stderr
            check_partition_fit(output, observations, "pitman_yor_gaussian_location")
            outputs.append(output)
        for filename in ("draws.csv", "allocations.csv", "diagnostics.csv"):
            assert (outputs[0] / filename).read_bytes() == (outputs[1] / filename).read_bytes()
        for index, (language, replacement, expected) in enumerate((
            ("python", ('discount = 0.4', 'discount = 1.0'), "discount"),
            ("R", ('method = "pitman_yor"', 'method = "hdp"'), "method must be"),
        ), 1):
            config.write_text(settings.replace(*replacement))
            output = directory / f"Invalid Pitman-Yor {index}"
            process = invoke(language, config, species_data, output, entrypoint=entrypoint)
            assert process.returncode != 0 and expected in process.stderr, process.stdout + process.stderr
            assert not output.exists()
        entrypoint = "examples/dirichlet_process/run.jl"
        for method, likelihood, observations in (
            ("neal2", "gaussian_location", [-1.2, -0.4, 0.0, 0.9, 1.3]),
            ("neal3", "gaussian_location", [-1.2, -0.4, 0.0, 0.9, 1.3]),
            ("neal8", "gaussian_location", [-1.2, -0.4, 0.0, 0.9, 1.3]),
            ("neal8", "poisson_lognormal", [0, 1, 2, 5, 8]),
        ):
            species_data.write_text("y\n" + "\n".join(map(str, observations)) + "\n")
            settings = (f'method = "{method}"\nlikelihood = "{likelihood}"\nstrength = 1.0\n'
                        'base_mean = 0.0\nbase_sd = 2.0\n' + common)
            if method == "neal8":
                settings += "auxiliaries = 3\n"
            config.write_text(settings)
            outputs = []
            for language in ("python", "R"):
                print(f"Real smoke: {language} -> Julia {method} {likelihood}", flush=True)
                output = directory / f"{method} {likelihood} {language}'s output"
                process = invoke(language, config, species_data, output, entrypoint=entrypoint)
                assert process.returncode == 0, process.stdout + process.stderr
                check_neal_fit(output, observations, likelihood, method)
                outputs.append(output)
            filenames = ["draws.csv", "allocations.csv", "diagnostics.csv"]
            if method != "neal3":
                filenames.append("parameters.csv")
            for filename in filenames:
                assert (outputs[0] / filename).read_bytes() == (outputs[1] / filename).read_bytes()
        for index, (language, invalid, expected) in enumerate((
            ("python", 'method = "neal2"\nlikelihood = "poisson_lognormal"\n', "require Neal 8"),
            ("R", 'method = "neal3"\nlikelihood = "poisson_lognormal"\n', "require Neal 8"),
            ("python", 'method = "neal8"\nauxiliaries = 0\n', "auxiliaries"),
            ("R", 'method = "neal8"\ndiscount = 0.3\n', "unsupported species-sampling settings"),
        ), 1):
            config.write_text(invalid + common)
            output = directory / f"Invalid Neal {index}"
            process = invoke(language, config, species_data, output, entrypoint=entrypoint)
            assert process.returncode != 0 and expected in process.stderr, process.stdout + process.stderr
            assert not output.exists()
        entrypoint = "examples/dp_slice/run.jl"
        observations = [-1.2, -0.4, 0.0, 0.9, 1.3]
        species_data.write_text("y\n" + "\n".join(map(str, observations)) + "\n")
        settings = ('method = "slice_efficient"\nstrength = 1.0\nobservation_sd = 0.7\n'
                    'base_mean = 0.0\nbase_sd = 2.0\n' + common)
        config.write_text(settings)
        outputs = []
        for language in ("python", "R"):
            print(f"Real smoke: {language} -> Julia efficient DP slice", flush=True)
            output = directory / f"DP slice {language}'s output"
            process = invoke(language, config, species_data, output, entrypoint=entrypoint)
            assert process.returncode == 0, process.stdout + process.stderr
            check_slice_fit(output, observations)
            outputs.append(output)
        for filename in ("draws.csv", "allocations.csv", "parameters.csv", "diagnostics.csv"):
            assert (outputs[0] / filename).read_bytes() == (outputs[1] / filename).read_bytes()
        for index, (language, extra, expected) in enumerate((
            ("python", 'discount = 0.3\n', "unsupported species-sampling settings"),
            ("R", 'likelihood = "poisson_lognormal"\n', "requires gaussian_location"),
            ("python", 'max_components = 20\n', "unsupported species-sampling settings"),
            ("R", 'betas = [1.0, 0.8]\n', "unsupported species-sampling settings"),
        ), 1):
            config.write_text(settings + extra)
            output = directory / f"Invalid DP slice {index}"
            process = invoke(language, config, species_data, output, entrypoint=entrypoint)
            assert process.returncode != 0 and expected in process.stderr, process.stdout + process.stderr
            assert not output.exists()
    print("Real Julia/Python/R smoke tests passed", flush=True)


if __name__ == "__main__":
    main()
