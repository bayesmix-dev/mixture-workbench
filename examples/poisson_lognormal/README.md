# Extend the workbench: a finite Poisson-lognormal mixture

This example adds a count likelihood and a Gaussian base measure on log rates. It is a complete source implementation with a fixed total number of components. The occupied count varies during sampling. The example retains the workbench's command-line and output conventions, so the existing R and Python wrappers can run it unchanged.

## Model and target

For observations $y_i$ in the nonnegative integers, the model is

$$
w \sim \operatorname{Dirichlet}(\gamma,\ldots,\gamma),  z_i\mid w\sim\operatorname{Categorical}(w),  y_i\mid z_i,\eta\sim\operatorname{Poisson}(\exp(\eta_{z_i})),  \eta_h\sim\operatorname{Normal}(\mu,s^2),  h=1,\ldots,M.
$$

`components` fixes $M$; `weight_shape` is the per-component parameter $\gamma$. The total Dirichlet concentration is $M\gamma$. `base_mean` and `base_sd` specify $\mu$ and $s$. These values are supplied in the configuration, rather than estimated from the observations. This is a finite mixture, not an MFM with a prior on the total component count.

After integrating out $w$, the sampler retains the labelled allocations $z$ and all $M$ log rates, including unused components. Write $n_h=\#\{i:z_i=h\}$. Up to the posterior normalising constant, its density is

$$
\frac{\Gamma(M\gamma)}{\Gamma(n+M\gamma)}\prod_{h=1}^M\frac{\Gamma(n_h+\gamma)}{\Gamma(\gamma)}\prod_{h=1}^M\phi(\eta_h;\mu,s^2)\prod_{i=1}^n\frac{\exp\left(y_i\eta_{z_i}-\exp(\eta_{z_i})\right)}{y_i!}.
$$

The base density is evaluated with respect to log rates $\eta_h$. No rate-transformation Jacobian enters this target.

A sweep first updates each allocation from its full conditional,

$$
\Pr(z_i=h\mid z_{-i},\eta,y)\propto(n_{h,-i}+\gamma)\operatorname{Poisson}\left(y_i;\exp(\eta_h)\right).
$$

For an occupied component, it then proposes $\eta'_h=\eta_h+\epsilon$, with $\epsilon\sim\operatorname{Normal}(0,\texttt{proposal\_sd}^2)$. The random-walk proposal is symmetric, so the Metropolis log ratio is

$$
\left(\eta'_h-\eta_h\right)\sum_{i:z_i=h}y_i-n_h\left(\exp(\eta'_h)-\exp(\eta_h)\right)-\frac{\left(\eta'_h-\mu\right)^2-\left(\eta_h-\mu\right)^2}{2s^2}.
$$

An empty component is refreshed directly from its Gaussian base measure. The Poisson likelihood and Gaussian log-rate base measure are nonconjugate; the component update therefore uses Metropolis sampling. This example implements these two updates without split-merge moves or tempering.

## Source changes and reuse

[PoissonLognormal.jl](PoissonLognormal.jl) contains the model, state, labelled target, allocation full conditional and component updates. Its sufficient statistics are component counts and sums of observations. It reuses `GeneralDPPMixtures.draw_logweights!` for allocation draws, `seedof` for separate chain streams, and `canonical_labels` for output.

[run.jl](run.jl) supplies an explicit entrypoint for this implemented model. It reuses `WorkbenchCLI` for numeric CSV parsing, command-line arguments, CSV writing, metadata writing and the core source digest. It validates its own configuration and count support. No existing Gaussian model or transition is changed.

To make a further extension, edit the likelihood and base-density calculations, then derive and implement the affected updates. The target and proposal checks in [the test file](../../test/poisson_lognormal.jl) show what must be established before using the result. A new TOML key alone cannot add a model feature: unknown keys and unsupported methods fail validation.

## Run from Julia, Python or R

Instantiate the repository environment first, as described in the [root README](../../README.md). From the repository root:

```sh
julia --startup-file=no --project=. examples/poisson_lognormal/run.jl \
  --config examples/poisson_lognormal/config.toml \
  --data examples/poisson_lognormal/counts.csv \
  --output results/poisson-julia

python3 interfaces/python/mixture_workbench.py \
  --entrypoint examples/poisson_lognormal/run.jl \
  --config examples/poisson_lognormal/config.toml \
  --data examples/poisson_lognormal/counts.csv \
  --output results/poisson-python

Rscript interfaces/R/run_sampler.R \
  --entrypoint examples/poisson_lognormal/run.jl \
  --config examples/poisson_lognormal/config.toml \
  --data examples/poisson_lognormal/counts.csv \
  --output results/poisson-r
```

Each output directory must be absent or empty. The supplied CSV is a small, synthetic count example. An input file must contain one header row and exactly one column of finite, nonnegative integer counts strictly smaller than $2^{53}$; the header name is arbitrary. The total count must also be strictly smaller than $2^{53}$. The shared CSV reader and component sufficient statistics use `Float64`; these bounds keep every count and count sum in its exact integer range. The flat TOML configuration accepts only `method`, `components`, `weight_shape`, `base_mean`, `base_sd`, `proposal_sd`, `iterations`, `warmup`, `thin`, `chains`, `seed` and `save_allocations`. `iterations` includes warmup, and retained iterations satisfy `(iteration - warmup) % thin == 0` after warmup.

The Python callable interface also accepts `entrypoint="examples/poisson_lognormal/run.jl"`; the R callable interface accepts `entrypoint = "examples/poisson_lognormal/run.jl"`. See [Python](../../interfaces/python/README.md) and [R](../../interfaces/R/README.md) for reading the result tables.

## Output labels and diagnostics

`draws.csv` contains `chain`, `iteration`, occupied count `k`, fixed total `M`, allocation `entropy`, `log_target` and `eta_1` through `eta_M`. For each draw, occupied labels are canonicalised by first appearance in the observation order. The log-rate columns follow the same order, with empty components appended. Thus `eta_cluster` matches each saved allocation in that draw. Component identities can change between draws.

`log_target` is the full labelled-state log density given above, including the Poisson factorials and normalised allocation and base densities, but excluding the posterior normalising constant. It does not include a canonical-partition multiplicity. After integrating all log rates, a partition with $k$ occupied groups receives contributions from $M!/(M-k)!$ labelled allocations. The exhaustive test checks that summation independently.

`allocations.csv` uses the standard long format and is omitted when `save_allocations=false`. `diagnostics.csv` reports Metropolis attempts and acceptances, empty-component refreshes, and an unassessed mixing entry. Counters include warmup. Metropolis acceptance measures the component proposals; it does not establish adequate exploration of partitions. `metadata.toml` and `metadata.csv` record effective model settings, seeds, input/configuration hashes, core and extension source hashes, Julia version and completion status.

## Validate the extension

```sh
julia --startup-file=no --project=. test/poisson_lognormal.jl
```

The standalone suite checks the likelihood against `Distributions.Poisson`, the full labelled target against an independent Dirichlet-multinomial calculation, allocation full conditionals, proposal symmetry and Metropolis ratios. It integrates the log rates by numerical quadrature for three observations and two components, sums all labelled allocations, and compares the resulting partition probabilities with sampled frequencies using batch means. Additional checks cover sufficient statistics, exact empty-component refreshes, canonical output, reproducibility, strict input rejection and the shared CSV protocol.

The quadrature comparison checks sampled partition frequencies against quadrature on that small problem. Assess mixing separately for each subsequent analysis, including the supplied count example.
