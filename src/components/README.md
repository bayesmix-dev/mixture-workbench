# Component calculations

This directory holds implemented observation/base combinations and the calculations they support.

| Source | Calculation |
|---|---|
| [gaussian_location.jl](gaussian_location.jl) | Normal observations with fixed variance and a Normal base for the mean |
| [poisson_lognormal.jl](poisson_lognormal.jl) | Poisson log-rate component used by Neal 8 |
| [gaussian_niw.jl](gaussian_niw.jl) | Multivariate NIW evidence used by MFM |
| [gaussian_fixed_location_iw.jl](gaussian_fixed_location_iw.jl) | Gaussian cluster statistics and covariance integration conditional on a location |

[interface.jl](interface.jl) defines the source-level species-sampling component operations. The collapsed sampler needs an analytic `logmarginal`; retained-atom methods need likelihood and base calculations plus an implemented atom update. Missing operations raise explicit errors. Supporting one operation does not enable every sampler.

Files are included into their existing owning modules: the Gaussian location component belongs to `SpeciesSampling`, the Poisson log-rate component to `NealSamplers`, the NIW evidence to `MFMTarget`, and the fixed-location IW calculations and `Cluster` type to `GeneralDPPMixtures`. This preserves their public identities while making component calculations easier to find. The last two implementations use their original Gaussian model/state types rather than the species-sampling component interface.

Start with [add-likelihood](../../recipes/add-likelihood.md) or [add-base-measure](../../recipes/add-base-measure.md), then identify the [model representation](../models/README.md) that will use the new calculations. The separate fixed-component Poisson example remains a complete worked implementation in [examples/poisson_lognormal](../../examples/poisson_lognormal/README.md).
