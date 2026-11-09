struct Initialization{D}
    c::Vector{Int}
    x::Vector{SVector{D,Float64}}
    S::Vector{Int}
end
function starting_configuration(m,k,chainid)
    rng=Xoshiro(seedof(m.seed,length(m.lower),201,chainid))
    c,x=kmeans_start(rng,m.y,k)
    # Starting configuration is shared, not sampled from the posterior/prior.
    forced=collect(1:k)
    for _ in 1:10000
        S=shell_draw(rng,m.spectrum,forced,0.0)
        F=feature_matrix(m,S,x)
        sv=svdvals(F)
        if minimum(sv)>1e-9*maximum(sv); return Initialization(c,x,S); end
        # Rare rank issue: enlarge initial forced support, within finite support.
        length(forced)<length(m.spectrum.kind) && push!(forced,length(forced)+1)
    end
    error("cannot construct full-rank initialization")
end
function initialize(m::Model{D,L},init::Initialization,algorithm::Symbol,seed) where {D,L}
    algorithm in (:conditional_notes,:marginal_collapsed) || throw(ArgumentError("unknown algorithm"))
    cl=[Cluster(m,x) for x in init.x]
    for i in eachindex(m.y); add!(cl[init.c[i]],m.y[i]); end
    for c in cl; refresh!(c,m); end
    streams=Streams(seed,length(m.y)); grow_streams!(streams,length(cl))
    b=MarginalState(m,copy(init.S),copy(init.c),cl,collect(eachindex(cl)),zeros(0,0),zeros(0,0),0.0,0.0,streams,Counters(),Float64[],Float64[],Float64[],[predictive_normalizer(m,n) for n in 0:length(m.y)])
    rebuild!(b)
    algorithm==:marginal_collapsed && return b
    s=ConditionalState(b,SVector{D,Float64}[],SMatrix{D,D,Float64,L}[],Float64[])
    reconstruct!(s)
end
function initialize(m,k,chainid,algorithm;seed=seedof(m.seed,length(m.lower),algorithm==:conditional_notes ? 301 : 302,chainid))
    initialize(m,starting_configuration(m,k,chainid),algorithm,seed)
end
