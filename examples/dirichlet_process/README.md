# Dirichlet-process mixtures: Neal 2, 3 and 8

This extension adds the three named kernels to the species-sampling interface. All three fit the same Gaussian location model introduced in the [Pitman–Yor example](../pitman_yor/README.md), with `DirichletProcess(strength)`. Neal 8 also fits the nonconjugate Poisson-lognormal component model. Strength and base parameters are fixed user settings.

## Kernels and targets

| Method | Retained variables | Allocation update |
|---|---|---|
| `neal2` | Partition and occupied component means | Existing components use count times likelihood; a birth uses strength times prior predictive and an exact single-observation posterior draw |
| `neal3` | Partition | Count times posterior predictive, with component means integrated out |
| `neal8` | Partition and occupied parameters | Existing components use count times likelihood; each of `auxiliaries = m` candidates has weight strength/m times likelihood |

For Neal 8, removing a singleton preserves its old parameter as one of the auxiliary candidates; only the remaining candidates are drawn from the base. Gaussian parameters are refreshed from their exact posterior. Poisson log rates use one symmetric random-walk Metropolis step per occupied component. These updates follow [Neal (2000), Algorithms 2, 3 and 8](https://glizen.com/radfordneal/ftp/mixmc.pdf).

For retained parameters, `log_target` evaluates

\[
\log p_{\mathrm{DP}}(\mathcal C)+\sum_{h=1}^k\log h(\theta_h)+\sum_i\log f(y_i\mid\theta_{z_i}).
\]

Its reference measure is counting measure on canonical partitions and Lebesgue measure on their occupied parameters. Weights are integrated out. Neal 3 instead reports the atom-integrated partition target. Gaussian Neal 2/8 draws also contain `partition_log_target`, which integrates the atoms and permits comparison of the same partition score across all three kernels. Absolute scores on the two reference measures differ.

The nonconjugate component model is \(y_i\mid\eta_{z_i}\sim\operatorname{Poisson}(e^{\eta_{z_i}})\), with \(\eta_h\sim N(\texttt{base_mean},\texttt{base_sd}^2)\). Its likelihood and log-rate update adapt the earlier [finite Poisson-lognormal example](../poisson_lognormal/README.md); the DP allocation law and occupied-state representation are new. Poisson MH ratios cancel factorial constants before evaluation and use `expm1` for small log-rate changes. Full Float64 joint densities can still lose absolute precision at very large counts; each count and their total must be below \(2^{53}\).

## Run and inspect

Use the same entry point and change only among implemented configurations:

```sh
julia --project=. examples/dirichlet_process/run.jl \
  --config examples/dirichlet_process/neal2.toml \
  --data examples/pitman_yor/data.csv --output results/neal2

Rscript interfaces/R/run_sampler.R \
  --entrypoint examples/dirichlet_process/run.jl \
  --config examples/dirichlet_process/neal3.toml \
  --data examples/pitman_yor/data.csv --output results/r-neal3

python3 interfaces/python/mixture_workbench.py \
  --entrypoint examples/dirichlet_process/run.jl \
  --config examples/dirichlet_process/poisson_neal8.toml \
  --data examples/poisson_lognormal/counts.csv --output results/python-neal8
```

[`neal8.toml`](neal8.toml) runs the Gaussian auxiliary sampler. `likelihood` is either `gaussian_location` (default) or `poisson_lognormal`; `method` must be explicit. Shared run/base settings have the [Pitman–Yor defaults](../pitman_yor/README.md). `auxiliaries` defaults to 3 and applies only to Neal 8. Gaussian `observation_sd` defaults to 1; Poisson `proposal_sd` defaults to 0.4. Neal 2/3 with Poisson observations, DP `discount`, split–merge, tempering, and model-inapplicable settings fail before output creation.

Neal 2/8 write `parameters.csv`: `chain,iteration,cluster,theta`. The `theta` value is a mean or log rate according to metadata `parameter_name`. Canonical cluster labels match `allocations.csv` within each draw; the wrappers expose the table as R `fit$parameters` and Python `fit.parameters`. Neal 3 has no retained parameter table. Here `M = k` records represented occupied atoms, with an infinite total disclosed in metadata.

## Source and evidence

[`src/models/species_sampling/algorithms/NealSamplers.jl`](../../src/models/species_sampling/algorithms/NealSamplers.jl) contains explicit transitions. To add a component model, implement its base/likelihood and a valid atom update; integrate it into the source-level capability checks and CLI if desired. Neal 2/3 additionally require exact conjugate operations. Unsupported operations produce errors rather than an inferred implementation.

[`test/neal_samplers.jl`](../../test/neal_samplers.jl) checks retained targets, singleton recycling, auxiliary weights, component conditionals and tiny partition posteriors using independent Gaussian vector densities and Poisson quadrature. It checks large-count MH differences against BigFloat references. The real wrapper smoke test reconstructs joint targets from exported allocations and parameters and compares seeded R/Python files.

Run `julia --project=. test/runtests.jl` and `python3 test/integration_smoke.py`. [Prompts and review decisions](prompts.md) and [executed validation](validation.json) accompany the implementation. Tiny reference checks do not assess exploration or speed on larger data.
