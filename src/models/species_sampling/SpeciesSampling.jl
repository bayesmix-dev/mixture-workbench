module SpeciesSampling

using Random, Distributions, SpecialFunctions

export PartitionPrior, DirichletProcess, PitmanYor, strength, discount,
    logeppf, seating_logweights, ComponentModel, GaussianLocation, observations,
    loglikelihood, logbase, randbase, logmarginal, randposterior, logpredictive,
    State, initialize, canonical_labels, cluster_counts, check_state,
    collapsed_allocation, collapsed_sweep!, collapsed_logtarget, sample_logweights

# Includes define the original public API inside this module. Directory
# boundaries describe responsibilities without adding module namespaces.
include("../../priors/partition/interface.jl")
include("../../priors/partition/dirichlet_process.jl")
include("../../priors/partition/pitman_yor.jl")
include("../../priors/partition/laws.jl")
include("../../components/interface.jl")
include("../../components/gaussian_location.jl")
include("collapsed_state.jl")
include("targets.jl")
include("algorithms/collapsed.jl")

end
