module GeneralDPPMixtures
using LinearAlgebra, Random, Distributions, SpecialFunctions, StaticArrays, Statistics
using Serialization, TOML, Printf
export Spectrum, Model, make_spectrum, simulate_problem, initialize, conditional_notes!, marginal_collapsed!, sweep!, run_chain, collapsed_logtarget, scalar_summary, canonical_labels, seedof, Execution, MarginalState, ConditionalState
include("spectrum.jl")
include("primitives.jl")
include("samplers.jl")
include("experiment.jl")
include("diagnostics.jl")
include("SpeciesSampling.jl")
include("NealSamplers.jl")
export SpeciesSampling, NealSamplers
end
