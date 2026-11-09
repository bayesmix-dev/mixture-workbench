# Mixture Workbench

A Julia research workbench for developing and checking Bayesian mixture samplers, with command-line interfaces for Julia, R and Python. Give an agent this repository and an extension task: it can follow a worked implementation, inspect its target and transitions, and run the relevant checks.

The starting implementations use Gaussian observations and include a mixture of finite mixtures (MFM), determinantal point process (DPP) mixtures and Strauss mixtures. The repository also contains model-specific split–merge, reallocation and likelihood-tempering implementations. The repulsive models retain coupled location priors; their split–merge implementations account for that dependence.

This is source code to adapt. A configuration selects an implemented sampler and its numerical settings. Adding a likelihood, base measure or hierarchical model requires implementing and validating the affected code; the configuration is not a modelling language.

## Run an example

Install Julia 1.12.3, then instantiate the checked-in environment:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. bin/run.jl \
  --config examples/gaussian/mfm.toml \
  --data examples/gaussian/data.csv \
  --output results/mfm
```

The synthetic dataset and short configurations in [`examples/gaussian/`](examples/gaussian/) exercise the public interface. Their iteration counts are for a smoke run; assess mixing before using posterior estimates.

Inputs have one header row and finite numeric values, with observations in rows. The Gaussian examples require more observations than dimensions and positive-definite sample covariance. Output directories must be absent or empty.

R and Python launch the same Julia executable. They require Julia and the instantiated repository environment. See [`interfaces/`](interfaces/) and the [command-line contract](docs/cli.md).

```sh
python3 interfaces/python/mixture_workbench.py \
  --config examples/gaussian/mfm.toml \
  --data examples/gaussian/data.csv \
  --output results/python-mfm

Rscript interfaces/R/run_sampler.R \
  --config examples/gaussian/mfm.toml \
  --data examples/gaussian/data.csv \
  --output results/r-mfm
```

## Find the relevant code

| Task | Start here |
|---|---|
| Run an existing implementation | [Command-line contract](docs/cli.md) |
| Understand the implemented distributions | [Targets](docs/targets/README.md) |
| Inspect split–merge proposals | [Split–merge moves](docs/algorithms/split-merge.md) |
| Inspect the tempering path | [Likelihood tempering](docs/algorithms/tempering.md) |
| Add a likelihood | [Recipe](recipes/add-likelihood.md) |
| Add a base measure | [Recipe](recipes/add-base-measure.md) |
| Add a different model | [Recipe](recipes/add-model.md) |
| Add a transition | [Recipe](recipes/add-move.md) |
| Check a change or investigate a run | [Validation](docs/validation.md), [mixing recipe](recipes/diagnose-mixing.md) |

[`AGENTS.md`](AGENTS.md) is the entry point for coding agents. [`CONTRIBUTING.md`](CONTRIBUTING.md) describes the evidence to include with a change.

For a complete extension, follow the [Poisson-lognormal finite-mixture showcase](examples/poisson_lognormal/README.md). It adds a nonconjugate count model through ordinary Julia source, keeps both wrappers unchanged, and records the [implementation and review prompts](examples/poisson_lognormal/prompts.md) and [validation evidence](examples/poisson_lognormal/validation.json).

The [Pitman–Yor extension](examples/pitman_yor/README.md) separates the partition prior from a conjugate Gaussian location model. Its zero-discount case gives a Dirichlet process; the example includes source-level interfaces, archived prompts and independent small-partition checks.

## What is shared

`src/GeneralDPPMixtures.jl` loads the Gaussian data structures, sufficient statistics, spectral representation and conditional/marginal DPP samplers. `src/Benchmark.jl` assembles additional model-specific modules, including the MFM baseline and point-process sampler. Module names are retained from the research code so that implementations and existing checks remain traceable.

The command-line layer chooses among these concrete implementations. It writes CSV draws and diagnostics plus configuration metadata. R and Python exchange files with this layer and do not implement an additional sampler.

Some retained `Benchmark` helpers belong to the original application driver: `data()` expects its application CSVs and `source_digest()` expects its research scripts. Those inputs are excluded from this repository. Use the documented CLI for portable runs and `WorkbenchCLI.source_digest()` for the publication source fingerprint.

The CLI uses fixed data-dependent prior settings and records them in metadata, including `empirical_bayes = true`. Its defaults and validation are in [`src/WorkbenchCLI.jl`](src/WorkbenchCLI.jl); see the [command-line contract](docs/cli.md) before fitting your own data.

Shared code follows demonstrated compatibility. For example, the NIW-collapsed MFM evidence is different from the fixed-location, covariance-collapsed evidence used in the DPP sampler. Changing one is not sufficient to change the other.

## Check the implementation

```sh
julia --project=. test/runtests.jl
python3 -m unittest discover -s test -p 'test_*.py'
Rscript test/test_wrappers.R
```

Tests include state invariants, independent target calculations and transition-specific balance checks. Passing these checks supports the calculations they exercise. Mixing assessment additionally needs independent runs, partition summaries and computational diagnostics; acceptance rates alone do not establish exploration. See [validation](docs/validation.md).

## Reproducibility and provenance

Commit the configuration, data-generation code, Julia environment and analysis commands used for a reported experiment. Keep large results outside Git. [`docs/provenance.json`](docs/provenance.json) records the source files used for this initial extraction.

To cite the software, use [`CITATION.cff`](CITATION.cff) and identify the commit used. The source is research software: individual implementations have different assumptions and validation coverage, recorded beside their algorithms and tests.

Released under the [BSD 3-Clause License](LICENSE).
