# Targets and state representations

The baseline observation models are multivariate Gaussian. The [worked count-model extension](../../examples/poisson_lognormal/README.md) adds a fixed-component Poisson mixture with Gaussian log-rate priors. The implementations below use different priors and different collapsed representations. Their targets must be identified before sharing a transition.

| Implementation | Retained state and integrated quantities | Source | Checks |
|---|---|---|---|
| MFM | Canonical partition, total component count `M`, Dirichlet parameter `gamma`, shifted-Poisson parameter `lambda`; means and covariances integrated under the NIW base measure | [src/MFMBaseline.jl](../../src/MFMBaseline.jl), [src/MFMTarget.jl](../../src/MFMTarget.jl) | [test/mfm_target.jl](../../test/mfm_target.jl), [test/mfm_hyperparameters.jl](../../test/mfm_hyperparameters.jl) |
| Marginal spectral DPP | Allocations, occupied locations and spectral subset; component covariances integrated conditional on locations | [src/primitives.jl](../../src/primitives.jl), [src/samplers.jl](../../src/samplers.jl) | [test/runtests.jl](../../test/runtests.jl), [test/dpp_sequential_balance.jl](../../test/dpp_sequential_balance.jl) |
| Conditional spectral DPP | The spectral model augmented with an unused configuration and component marks used by the conditional updates | [src/samplers.jl](../../src/samplers.jl) | [test/runtests.jl](../../test/runtests.jl) |
| General point-process sampler | Allocations, occupied and unused locations, component marks and auxiliary variable `logu`; location prior supplied by DPP or Strauss geometry | [src/PointProcessSampler.jl](../../src/PointProcessSampler.jl) | [test/runtests.jl](../../test/runtests.jl), [test/mahalanobis_strauss.jl](../../test/mahalanobis_strauss.jl) |

The Poisson-lognormal extension retains labelled allocations and all fixed-`M` log rates, integrating only the Dirichlet weights. Its complete target and output representation are documented in the [example](../../examples/poisson_lognormal/README.md), with independent checks in [test/poisson_lognormal.jl](../../test/poisson_lognormal.jl).

`k` is the occupied component count. `M` is the total component count in the MFM and general point-process states; in the spectral DPP states the corresponding count is the size of the retained spectral subset. Numerical equality of `k` or `M` across models does not identify equal posterior distributions.

## MFM

`MFMTarget.log_marginal` integrates both component means and covariances. `partition_logtarget` adds the Dirichlet allocation factor and the falling factorial counting injections of occupied blocks into component labels. `logtarget` additionally includes `M - 1 | lambda ~ Poisson(lambda)` and the exponential hyperpriors used by this implementation. See the source docstrings for the precise reference measure and normalising terms.

The MFM initializer uses the data mean for `mu0` and sets `kappa = 0.1`. The CLI sets the covariance-prior scale from the sample covariance and records `empirical_bayes = true`. These fixed data-dependent settings are part of the implemented model. The MFM samples `gamma`; the repulsive-model `weight_shape` option does not fix that parameter.

## Repulsive mixtures

The DPP and Strauss location priors couple component locations. Their observation likelihood is Gaussian and some covariance calculations remain analytic. This does not make the location prior a product base measure. A move that updates locations must include the joint geometry contribution.

`PointProcessSampler.DPPPrior` uses the finite spectral ensemble from the model's eigenvalues. `StraussPrior` uses an activity, interaction radius and log interaction parameter. `geometry` and `edit` evaluate the corresponding location-prior contribution. Geometry densities use their documented point-process reference measure; do not compare them with partition densities without accounting for the representation.

## Before extending a target

Record the observation space, model parameters, allocation or hierarchy law, retained variables, integrated variables, support and reference measure. Identify which computations are target evaluations and which only construct a proposal. Add an independent small-case calculation before reusing a transition.

See [likelihood tempering](../algorithms/tempering.md) for the intermediate distributions and [validation](../validation.md) for the scope of existing evidence.
