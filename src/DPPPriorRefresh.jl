module DPPPriorRefresh

using GeneralDPPMixtures, Random
const G=GeneralDPPMixtures

export refresh!

function candidate(s::G.MarginalState,rng,maxtries)
    m=s.model; n=length(m.y)
    n>0 || throw(ArgumentError("prior refresh requires observations"))
    # This is the original finite DPP prior conditioned on being nonempty,
    # without conditioning on an occupied count or a previous spectrum.
    S=G.prior_support(rng,m.spectrum;minimum=1)
    counters=deepcopy(s.counters)
    xs=G.palm_sample(rng,m,S,typeof(first(m.y))[],counters;maxtries)
    logweights=[G.loggamma_draw(rng,m.a) for _ in S]
    weights=exp.(logweights.-maximum(logweights))
    full_labels=[G.draw_weights(rng,weights) for _ in 1:n]
    # First-appearance orientation makes locations and canonical memberships
    # agree even when several full-configuration components are unused.
    occupied=unique(full_labels)
    canonical=zeros(Int,length(S))
    for (h,id) in enumerate(occupied); canonical[id]=h; end
    labels=[canonical[id] for id in full_labels]
    clusters=[G.Cluster(m,xs[id]) for id in occupied]
    for i in eachindex(labels); G.add!(clusters[labels[i]],m.y[i]); end
    for c in clusters; G.refresh!(c,m,s.predictive_norms); end
    active=collect(eachindex(clusters))
    proposed=G.MarginalState(m,S,labels,clusters,active,zeros(0,0),zeros(0,0),0.,0.,
        s.streams,counters,Float64[],Float64[],Float64[],s.predictive_norms)
    G.rebuild!(proposed)
    proposed
end

"""
    refresh!(s; rng=s.streams.spectral, maxtries=10^7, observer=nothing)

Replace `s` by an independent draw from the beta=0 prior endpoint. This kernel
must only be used where the collapsed data likelihood has power zero.

Draw the nonempty spectrum, its full projection DPP configuration, symmetric
Dirichlet weights and iid allocations, then retain occupied locations. No
factorial correction is added: the ordered DPP density contributes 1/M!, the
choice of occupied labels contributes M!/(M-k)!, and integrating the unused
locations contributes (M-k)! to the occupied determinant.

All fallible draws/cache construction use a private RNG copy and candidate
state. The optional observer receives `(old, candidate)` before commit. On
failure, the original state and RNG are unchanged. Existing RNG stream objects
and their sequences are preserved; only `rng` advances and new location streams
are appended if the occupied count exceeds previous storage capacity.
"""
function refresh!(s::G.MarginalState;rng::AbstractRNG=s.streams.spectral,
        maxtries::Integer=10^7,observer=nothing)
    maxtries>=1 || throw(ArgumentError("maxtries must be positive"))
    # copy(TaskLocalRNG()) materializes an independent Xoshiro state, whereas
    # deepcopy(TaskLocalRNG()) would still refer to the live task's RNG.
    trialrng=copy(rng)
    hasmethod(copy!,Tuple{typeof(rng),typeof(trialrng)}) ||
        throw(ArgumentError("prior refresh requires an RNG supporting copy!"))
    proposed=candidate(s,trialrng,maxtries)
    observer===nothing || observer(s,proposed)
    # No probability evaluation or fallible numerical work follows this point.
    # Preserve all existing stream identities and append deterministic streams
    # only when the new occupied configuration needs additional slots.
    G.grow_streams!(s.streams,length(proposed.clusters))
    copy!(rng,trialrng)
    s.S=proposed.S; s.c=proposed.c; s.clusters=proposed.clusters; s.active=proposed.active
    s.F=proposed.F; s.Q=proposed.Q; s.ld=proposed.ld; s.bound=proposed.bound
    s.counters=proposed.counters
    s.workf=proposed.workf; s.workv=proposed.workv; s.workw=proposed.workw
    s
end

end
