mutable struct MarginalState{D,L}
    model::Model{D,L}
    S::Vector{Int}
    c::Vector{Int}
    clusters::Vector{Cluster{D,L}}
    active::Vector{Int}
    F::Matrix{Float64}
    Q::Matrix{Float64}
    ld::Float64                    # dimensionless determinant
    bound::Float64
    streams::Streams
    counters::Counters
    workf::Vector{Float64}
    workv::Vector{Float64}
    workw::Vector{Float64}
    predictive_norms::Vector{Float64}
end
mutable struct ConditionalState{D,L}
    base::MarginalState{D,L}
    unused::Vector{SVector{D,Float64}}
    covarianceL::Vector{SMatrix{D,D,Float64,L}} # active order then unused
    weights::Vector{Float64}
end
core(s::MarginalState)=s
core(s::ConditionalState)=s.base
