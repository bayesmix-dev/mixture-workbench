# Pitman–Yor Gaussian location mixture

This extension changes the allocation prior through ordinary Julia source. [`SpeciesSampling`](../../src/SpeciesSampling.jl) separates a partition prior from a component model. The runnable example uses a Normal base distribution and a known observation standard deviation. R and Python use the existing file interface.

## Model and target

Let (G\sim\mathrm{PY}(d,\alpha,H)), with (H=N(m_0,s_0^2)), and let (y_i\mid\theta_i\sim N(\theta_i,\sigma^2)), independently given component means sampled from (G). The settings `strength`, `discount`, `base_mean`, `base_sd` and `observation_sd` specify \(\alpha,d,m_0,s_0,\sigma\). These parameters are fixed and supplied by the user.

The supported range is \(0\le d<1\), with \(\alpha>0\) for \(d=0\) and \(\alpha>-d\) otherwise. For a partition \(\mathcal C\) with block sizes \(n_1,\ldots,n_k\), its probability is

\[
p(\mathcal C)=\frac{\prod_{j=1}^{k-1}(\alpha+jd)}{\prod_{i=1}^{n-1}(\alpha+i)}\prod_{h=1}^k\prod_{r=1}^{n_h-1}(r-d).
\]

Empty products equal one. The first observation starts a block with probability one, including when strength is negative. Zero discount recovers the Dirichlet-process partition law. See [Pitman (2006), §3.2, Theorem 3.2](https://www.stat.berkeley.edu/~aldous/206-Exch/Papers/pitman_CSP.pdf).

The state retains the partition and integrates out both (G) and the component means. Its log target is

\[
\log p(\mathcal C)+\sum_{C\in\mathcal C}\log\int\prod_{i\in C}N(y_i;\theta,\sigma^2)\,N(\theta;m_0,s_0^2)\,d\theta.
\]

This density uses counting measure on set partitions and Lebesgue measure on observations. An allocation sweep removes one observation, assigns existing blocks weights \((n_h-d)\) times their posterior predictive density, and assigns a new block weight \(\alpha+kd\) times the prior predictive density. Means are integrated analytically.

## Run from Julia, R or Python

From the repository root:

```sh
julia --project=. examples/pitman_yor/run.jl \
  --config examples/pitman_yor/config.toml \
  --data examples/pitman_yor/data.csv --output results/pitman-yor

Rscript interfaces/R/run_sampler.R \
  --entrypoint examples/pitman_yor/run.jl \
  --config examples/pitman_yor/config.toml \
  --data examples/pitman_yor/data.csv --output results/r-pitman-yor

python3 interfaces/python/mixture_workbench.py \
  --entrypoint examples/pitman_yor/run.jl \
  --config examples/pitman_yor/config.toml \
  --data examples/pitman_yor/data.csv --output results/python-pitman-yor
```

The callable wrappers accept the same `entrypoint` argument. Input has exactly one numeric column. Constant observations and a single observation are supported because the observation variance is specified explicitly.

## Configuration and output

`method = "pitman_yor"` selects this implementation. The model settings are `strength = 1.0`, `discount = 0.3`, `observation_sd = 1.0`, `base_mean = 0.0`, `base_sd = 2.0` by default. The run settings default to `initial_clusters = 1`, `iterations = 4000`, `warmup = 1000`, `thin = 5`, `chains = 4`, `seed = 123`, `save_allocations = true`. Labels initially cycle through `initial_clusters` occupied blocks.

Unknown keys, invalid parameter values and unsupported settings fail before an output directory is created. Split–merge and tempering need transitions derived for this target; their settings are rejected. The known variance and Normal base are implemented source choices. A new name in TOML does not create a likelihood or base measure.

The shared CSV/TOML outputs preserve chain, iteration and observation identifiers. Allocations are canonical within a draw. Here `M = k` counts represented occupied atoms; metadata records `total_components = "infinite"` and `M_definition`. No finite truncation is used. `log_target` is the partition target above. Exact Gibbs assignment counters include warmup; a counter value is not a mixing diagnostic.

## Extend and check

Implement a Julia subtype of `ComponentModel` and its likelihood/base methods. A collapsed kernel additionally needs `logmarginal`; methods that draw component means need `randposterior`. Unsupported operations raise an error. A different partition prior implements its EPPF and seating law. Keep these mathematical contracts aligned and add an independent reference calculation.

[`test/species_sampling.jl`](../../test/species_sampling.jl) checks partition normalization, seating ratios, the DP limit, Gaussian evidence and tiny posterior calculations. [`test/species_sampling_cli.jl`](../../test/species_sampling_cli.jl) checks validation and output semantics. The real R/Python smoke test reconstructs saved targets using sequential predictive densities and compares seeded outputs from both wrappers.

Run `julia --project=. test/runtests.jl` and `python3 test/integration_smoke.py`. The archived [prompts](prompts.md) explain the implementation and review workflow; [validation.json](validation.json) records executed checks and their scope. Small reference problems check the implemented calculations and transitions. Exploration on larger datasets remains unassessed.
