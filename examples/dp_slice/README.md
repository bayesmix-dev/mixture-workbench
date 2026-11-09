# Efficient slice sampling for a Dirichlet-process mixture

This implementation fits the same fixed-strength Gaussian DP location mixture as the [Neal examples](../dirichlet_process/README.md). It retains an ordered stick-breaking representation and uses the dependent slice-efficient update of [Kalli, Griffin and Walker (2011), Section 2](https://kar.kent.ac.uk/24721/1/slice_mix.pdf). The earlier implementation was located in `HSNCP-consistency`; the [provenance record](provenance.json) identifies the exact source audited. The workbench implementation is newly written and shares its component model with the other DP kernels.

## State and transition

The weights are \(w_j=V_j\prod_{\ell<j}(1-V_\ell)\), with independent \(V_j\sim\mathrm{Beta}(1,\alpha)\). Internal allocations retain these ordered indices, including empty gaps. One sweep:

1. Draw occupied means from their Gaussian posterior and empty means from the base.
2. Integrate out old slices and draw \(V_j\mid z\sim\mathrm{Beta}(1+n_j,\alpha+\sum_{\ell>j}n_\ell)\).
3. Draw \(u_i\mid w,z\sim\mathrm{Uniform}(0,w_{z_i})\), extending prior sticks and atoms until residual mass is below \(\min_i u_i\).
4. Draw allocations with weights \(1\{u_i<w_j\}f(y_i\mid\theta_j)\).

The residual bound excludes every later atom. The finite prefix changes with the slices; no fixed truncation is introduced. See the [published article](https://doi.org/10.1007/s11222-009-9150-y) for the blocked update.

[`src/models/species_sampling/algorithms/EfficientSlice.jl`](../../src/models/species_sampling/algorithms/EfficientSlice.jl) computes stick complements, weights, slices and residuals in log scale. Shape-one sticks use an exponential construction; other Beta sticks use log Gamma variates with a shape boost for a small second shape. Unrepresentable draws or residual updates abort explicitly. Configuration has no component cap or residual tolerance that changes the target.

## Run from any supported language

```sh
julia --project=. examples/dp_slice/run.jl \
  --config examples/dp_slice/config.toml \
  --data examples/pitman_yor/data.csv --output results/dp-slice

Rscript interfaces/R/run_sampler.R \
  --entrypoint examples/dp_slice/run.jl \
  --config examples/dp_slice/config.toml \
  --data examples/pitman_yor/data.csv --output results/r-dp-slice

python3 interfaces/python/mixture_workbench.py \
  --entrypoint examples/dp_slice/run.jl \
  --config examples/dp_slice/config.toml \
  --data examples/pitman_yor/data.csv --output results/python-dp-slice
```

`method = "slice_efficient"` is required; `likelihood = "gaussian_location"` is the implemented component model. Strength, observation scale, Normal base and run settings use the [DP example defaults](../dirichlet_process/README.md). Strength is fixed and positive. This entry point rejects Pitman–Yor discount, other likelihoods, auxiliary-candidate counts, component caps, split–merge and tempering before creating output.

## Output representations

Allocations are canonical copies for convenient partition summaries. Internal GEM indices remain unchanged. `parameters.csv` contains occupied `chain,iteration,cluster,theta,gem_index`; its canonical `cluster` joins to allocations, and `gem_index` identifies the ordered atom. R `fit$parameters` and Python `fit.parameters` expose this table.

`k` counts occupied atoms. `M` counts the entire materialized ordered prefix, including empty gaps and unused tail atoms. Metadata states this meaning and records an infinite total. Neither canonical labels nor GEM indices identify a component across independent draws or chains.

`log_target` is the canonical partition score with all weights and means integrated out, as in Neal 3. It evaluates a projection of the slice state; metadata records `log_target_is_projection = true`. The additional `ordered_log_target` retains the original ordered allocation indices and occupied means, integrating sticks, slices and empty means:

\[
\sum_{j=1}^{\max z}\left[\log\alpha+\log B\left(1+n_j,\alpha+\sum_{\ell>j}n_\ell\right)\right]+\sum_{j:n_j>0}\log h(\theta_j)+\sum_i\log f(y_i\mid\theta_{z_i}).
\]

This second score uses counting measure on ordered allocation vectors and Lebesgue measure on occupied means. Empty gaps contribute Beta factors. The full augmented density also includes the sweep auxiliaries. Diagnostics count allocation updates and stick draws over warmup and sampling; the materialized prefix length is an execution quantity, not a convergence diagnostic.

## Validation and extension

[`test/efficient_slice.jl`](../../test/efficient_slice.jl) checks independent Gaussian vector evidence, tiny exact partition posteriors, numerical Beta integrals, posterior stick shapes and moments, component draws, eligibility, tail stopping, ordered gaps, very small strength and reproducibility. The CLI tests and actual R/Python smoke runs check both projected scores from exported allocations and parameters, and compare seeded files between wrappers.

Run `julia --project=. test/runtests.jl` and `python3 test/integration_smoke.py`. [Prompts and workflow](prompts.md), [source provenance](provenance.json), and [executed validation](validation.json) accompany the implementation. Extension to another component model requires its valid parameter update and corresponding checks. Tiny reference agreement does not assess mixing or speed for larger datasets.
