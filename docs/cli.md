# Command-line contract

The Julia entry point runs an implemented model. R and Python pass paths to this entry point and read its files. They do not generate a model or choose a sampling algorithm.

```sh
julia --project=. bin/run.jl --config FILE.toml --data FILE.csv --output DIR
julia --project=. bin/run.jl --help
```

The input is an unquoted numeric CSV with one header row, nonempty unique column names and one observation per row. Every value must be finite. The Gaussian implementation treats columns as coordinates, requires `n > max(d, 2)` and requires a positive-definite empirical covariance. Degenerate data fail validation; the engine does not add covariance jitter. Use `examples/gaussian/data.csv` as the format reference.

## Configuration

The flat TOML configuration accepts the keys below. [`src/WorkbenchCLI.jl`](../src/WorkbenchCLI.jl) contains the authoritative defaults and validation. Prior-specific keys apply only to the corresponding implemented models.

| Key | Default | Meaning |
|---|---|---|
| `method` | `"mfm"` | One of the concrete implementations below |
| `iterations` | `1000` | Total sweeps, including warmup |
| `warmup` | `500` | Discarded initial sweeps |
| `thin` | `5` | Retained-draw spacing after warmup |
| `chains` | `2` | Independent chains or independent ladders |
| `seed` | `123` | Input for derived independent random streams |
| `initial_clusters` | `2` | Initial occupied count |
| `covariance_scale` | `1.0` | Multiplier of empirical covariance in the covariance prior |
| `weight_shape` | `0.5` | Repulsive-mixture weight shape; MFM samples its own `gamma` |
| `spectrum_target` | `3.0` | DPP spectral construction target |
| `spectrum_c0` | `0.9` | DPP leading spectral value |
| `spectrum_cutoff` | `0.02` | DPP spectral retention cutoff |
| `strauss_activity` | `3.0` | Strauss activity parameter used by this implementation |
| `strauss_radius` | `0.5` | Strauss interaction radius |
| `strauss_log_interaction` | `-2.0` | Log Strauss interaction parameter |
| `save_allocations` | `true` | Write retained observation allocations |
| `split_merge` | `false` | Optional joint sequential MFM split–merge moves |
| `betas` | `[1.0]` | Likelihood-power ladder |
| `swap_every` | `5` | Sweeps between exchange attempts |

Retained draws satisfy `iteration > warmup` and `(iteration - warmup) % thin == 0`. Settings must leave at least one retained draw. A ladder has strictly descending powers in `[0, 1]` starting at `1`; the single-power default runs without tempering. Only `mfm` and `dpp_marginal` support the CLI likelihood-tempering option.

| `method` | Implemented sampler |
|---|---|
| `mfm` | NIW-collapsed Gaussian mixture of finite mixtures |
| `dpp_conditional` | Conditional spectral DPP sampler |
| `dpp_marginal` | Marginal spectral DPP sampler |
| `dpp_marginal_sm` | Marginal DPP sampler with the existing DPP split–merge move |
| `dpp_general` | General point-process sampler using DPP geometry |
| `strauss_general` | General point-process sampler using Strauss geometry |

Unknown configuration keys and unsupported combinations fail before sampling. Explicit `spectrum_*` settings are supported only by DPP methods, `weight_shape` by repulsive methods, and `strauss_*` settings by `strauss_general`. The `split_merge` flag enables only MFM moves; select `dpp_marginal_sm` for the existing DPP split–merge implementation.

A new likelihood or model requires source changes, checks and an explicit CLI path. Adding its name to TOML does not implement it.

## Data-dependent prior settings

The current CLI uses fixed data-dependent prior settings and records `empirical_bayes = true`. The covariance-prior scale is `covariance_scale` times the sample covariance and its degrees of freedom are `d + 2`. The MFM uses the sample mean for its NIW location centre and `kappa = 0.1`; it samples `gamma` and `lambda` with independent exponential hyperpriors of rate `0.1`. `weight_shape` does not fix MFM `gamma`.

The repulsive models use the empirical padded location domain constructed in `Benchmark.model`. These choices are recorded in metadata. They form part of the fitted model and should be considered when comparing results or changing prior specifications.

## Outputs

The output path must be absent or an existing empty directory. Nonempty directories are refused. A run writes:

| File | Contents |
|---|---|
| `draws.csv` | `chain`, `iteration`, `k`, `M`, `entropy`, `log_target` |
| `allocations.csv` | Optional `chain`, `iteration`, `observation`, `cluster` rows |
| `diagnostics.csv` | `chain`, `metric`, `attempted`, `accepted`, `value`, `status` |
| `transport.csv` | With tempering: retained `chain`, `iteration`, `slot`, `beta`, `replica` mappings |
| `config.toml` | Resolved configuration, including defaults |
| `metadata.toml` | Run settings and provenance in a structured form |
| `metadata.csv` | Metadata readable by the dependency-free wrappers |

`log_target` records the cold collapsed target for MFM and spectral DPP methods. For `dpp_conditional`, this is the marginal density of its base projection, rather than the full augmented-state density. It is unavailable and written as `NaN` for `dpp_general` and `strauss_general`. Reference measures and integrated variables differ between models; absolute log targets across methods are not model-comparison scores.

Allocations use canonical labels within each draw. Those labels are local to the draw; preserve observation, chain and iteration identifiers when computing partition summaries. With tempering, retained posterior draws come from the target-power slot. `transport.csv` records replica identities across all slots at retained iterations.

Diagnostics count attempts and acceptances over all sweeps, including warmup and, for split–merge, all replicas. Round trips also count the full run; the thinned transport file is not the event history from which they were calculated. Metadata records `running`, `completed` or `failed` execution status. Completion describes execution and does not establish convergence. A CLI error prints an explanation and exits with status `1`.

CSV files are the interoperability boundary. Wrappers provide access to recorded quantities; they do not add convergence guarantees or silently relabel components.

## R

The R wrapper uses base R:

```r
source("interfaces/R/mixture_workbench.R")
fit <- run_sampler(
  "examples/gaussian/mfm.toml",
  "examples/gaussian/data.csv",
  "results/r-mfm"
)
head(fit$draws)
fit$diagnostics
summary(fit)

# Read completed files without starting Julia.
fit <- read_sampler_output("results/r-mfm")
```

The command-line equivalent is `Rscript interfaces/R/run_sampler.R --config FILE.toml --data FILE.csv --output DIR`. `summary(fit)` gives marginal numerical summaries. See [`interfaces/R/README.md`](../interfaces/R/README.md) for return types and error handling.

## Python

The Python wrapper uses the standard library:

```python
import sys
sys.path.insert(0, "interfaces/python")
from mixture_workbench import run_sampler, read_output

fit = run_sampler(
    "examples/gaussian/mfm.toml",
    "examples/gaussian/data.csv",
    "results/python-mfm",
)
print(fit.draws[:3])
print(fit.diagnostics)
fit = read_output("results/python-mfm")
```

The command-line equivalent is `python3 interfaces/python/mixture_workbench.py --config FILE.toml --data FILE.csv --output DIR`. See [`interfaces/python/README.md`](../interfaces/python/README.md) for return types and error handling.

Both wrappers support `--julia EXEC` and `--project-root ROOT`, with corresponding callable arguments. They run synchronously and preserve nonzero engine exit status at the command line.

`--entrypoint SCRIPT` selects a different implemented Julia entry point, with a corresponding `entrypoint` callable argument. A relative script path is resolved within the project root. An extension may use this route while retaining the same input/output contract; supplying a script path does not generate its implementation.

## Pitman–Yor example entry point

Use `--entrypoint examples/pitman_yor/run.jl` through either wrapper, or run that script with Julia directly. Its strict configuration and Gaussian location model are documented in the [example](../examples/pitman_yor/README.md). Its single-column observations and fixed positive observation variance allow even one observation or constant data. This entry point uses user-specified priors and writes `M = k` for represented occupied atoms, with the infinite total recorded in metadata.

## Dirichlet-process entry point

Use `--entrypoint examples/dirichlet_process/run.jl` with explicit `method = "neal2"`, `"neal3"` or `"neal8"`. See the [DP example](../examples/dirichlet_process/README.md) for strict model-specific settings. Neal 2/8 write an optional numeric `parameters.csv` (`chain,iteration,cluster,theta`), exposed by both wrappers; labels match allocations within a draw. Neal 3 integrates the parameters out. `log_target` has the retained joint or collapsed partition reference measure stated in metadata; Gaussian retained runs also write the atom-integrated `partition_log_target`.
