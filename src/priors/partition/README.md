# Partition priors

[interface.jl](interface.jl) defines the `PartitionPrior` interface. [dirichlet_process.jl](dirichlet_process.jl) and [pitman_yor.jl](pitman_yor.jl) define the concrete priors and validate their supported parameters. [laws.jl](laws.jl) supplies the EPPF and seating-weight calculations.

These files are included by `GeneralDPPMixtures.SpeciesSampling`; the directory introduces no additional module namespace. Observation likelihoods and allocation state are separate from these laws.

The [species-sampling family](../../models/species_sampling/README.md) combines these priors with concrete component models. Its collapsed Gaussian sweep supports DP and PYP priors. The named Neal kernels and current slice sampler impose their own DP restrictions; a shared prior interface does not broaden those implementations automatically.
