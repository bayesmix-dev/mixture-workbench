"""Run an implemented Julia sampler and read its documented CSV outputs.

This module has no third-party Python dependencies. A Julia executable and the
repository's instantiated Julia environment are required for actual sampling.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any


@dataclass
class SamplerResult:
    """Tables from a completed run; component labels are local to each draw."""

    output: Path
    draws: list[dict[str, int | float]]
    diagnostics: list[dict[str, Any]]
    metadata: dict[str, str]
    allocations: list[dict[str, int | float]] | None = None


def _read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Inconsistent CSV row width: {path}")
    return list(reader.fieldnames), rows


def _numeric_table(path: Path, required: set[str]) -> list[dict[str, int | float]]:
    fields, rows = _read_table(path)
    missing = required.difference(fields)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {', '.join(sorted(missing))}")
    integer_fields = {"chain", "iteration", "k", "M", "observation", "cluster"}
    converted = []
    for row in rows:
        converted_row = {}
        for key, value in row.items():
            try:
                converted_row[key] = int(value) if key in integer_fields else float(value)
            except ValueError as error:
                raise ValueError(f"Invalid numeric value for {key!r} in {path}") from error
        converted.append(converted_row)
    return converted


def _diagnostic_value(value: str) -> str | int | float:
    for convert in (int, float):
        try:
            return convert(value)
        except ValueError:
            pass
    return value


def read_output(output: str | Path) -> SamplerResult:
    """Read an existing completed run, without starting Julia."""
    output_path = Path(output).expanduser().resolve()
    metadata_fields, metadata_rows = _read_table(output_path / "metadata.csv")
    if not {"key", "value"}.issubset(metadata_fields):
        raise ValueError("metadata.csv must contain key and value columns")
    metadata = {row["key"]: row["value"] for row in metadata_rows}
    if metadata.get("schema_version") != "1":
        raise ValueError(f"Unsupported output schema_version: {metadata.get('schema_version')!r}")
    if metadata.get("run_status") != "completed":
        raise ValueError(f"Run is not completed: {metadata.get('run_status')!r}")
    draws = _numeric_table(
        output_path / "draws.csv", {"chain", "iteration", "k", "M", "entropy", "log_target"}
    )
    _, diagnostic_rows = _read_table(output_path / "diagnostics.csv")
    diagnostics = [
        {key: _diagnostic_value(value) for key, value in row.items()}
        for row in diagnostic_rows
    ]
    allocation_path = output_path / "allocations.csv"
    allocations = (
        _numeric_table(allocation_path, {"chain", "iteration", "observation", "cluster"})
        if allocation_path.exists()
        else None
    )
    return SamplerResult(output_path, draws, diagnostics, metadata, allocations)


def run_sampler(
    config: str | Path,
    data: str | Path,
    output: str | Path,
    julia: str | Path = "julia",
    project_root: str | Path | None = None,
    entrypoint: str | Path | None = None,
) -> SamplerResult:
    """Run Julia synchronously and return its output tables.

    ``config`` is an existing TOML configuration for an implemented model;
    ``data`` is a numeric CSV with a header. No model language is interpreted.
    A failed subprocess raises ``subprocess.CalledProcessError`` with its exit
    status. Arguments are passed directly, without a shell.
    """
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    config_path = Path(config).expanduser().resolve()
    data_path = Path(data).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    runner = Path(entrypoint).expanduser() if entrypoint is not None else Path("bin/run.jl")
    runner = (runner if runner.is_absolute() else root / runner).resolve()
    for path in (config_path, data_path, runner):
        if not path.is_file():
            raise FileNotFoundError(path)
    command = [
        str(julia),
        "--startup-file=no",
        f"--project={root}",
        str(runner),
        "--config", str(config_path),
        "--data", str(data_path),
        "--output", str(output_path),
    ]
    subprocess.run(command, check=True, cwd=root)
    return read_output(output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Existing TOML configuration")
    parser.add_argument("--data", required=True, help="Numeric CSV with a header")
    parser.add_argument("--output", required=True, help="Destination directory")
    parser.add_argument("--julia", default="julia", help="Julia executable or command")
    parser.add_argument("--project-root", help="Repository root (defaults to this checkout)")
    parser.add_argument("--entrypoint", help="Implemented Julia script, relative to repository root")
    args = parser.parse_args(argv)
    try:
        fit = run_sampler(
            args.config, args.data, args.output, args.julia, args.project_root, args.entrypoint
        )
    except subprocess.CalledProcessError as error:
        print(f"Sampler failed with exit status {error.returncode}", file=sys.stderr)
        return error.returncode if error.returncode > 0 else 128 - error.returncode
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"Read {len(fit.draws)} draws from {fit.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
