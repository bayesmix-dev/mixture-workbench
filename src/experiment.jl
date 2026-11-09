function simulate_problem(d::Int,family::Symbol,seed::Integer; nper=100,spectrum=make_spectrum(d))
    family in (:gaussian,:student) || throw(ArgumentError("family must be gaussian or student"))
    rng=Xoshiro(seedof(seed,d,family==:gaussian ? 101 : 102))
    simulate_problem(Val(d),family,seed,nper,spectrum,rng)
end
function simulate_problem(::Val{D},family,seed,nper,spectrum,rng) where D
    mus=[SVector{D,Float64}(ntuple(e->e<=cld(D,2) ? -4.0 : 4.0,D)),zeros(SVector{D,Float64}),4ones(SVector{D,Float64})]
    y=SVector{D,Float64}[]; truth=Int[]
    for h in 1:3, _ in 1:nper
        z=SVector{D,Float64}(ntuple(_->randn(rng),D))
        family==:student && (z/=sqrt(rand(rng,Chisq(6))/6))
        push!(y,mus[h]+z); push!(truth,h)
    end
    center=mean(y)
    zeta=SVector{D,Float64}(ntuple(e->maximum(abs(v[e]-center[e]) for v in y),D))
    Model(y,truth,center-2.5zeta,center+2.5zeta,spectrum;seed=seed)
end
"""Deterministic-seeded k-means++ with Lloyd updates and empty-cluster repair."""
function kmeans_start(rng,y,k)
    k<=length(y) || throw(ArgumentError("k exceeds data size"))
    centers=[y[rand(rng,eachindex(y))]]
    for _ in 2:k
        w=[minimum(sum(abs2,v-x) for x in centers) for v in y]
        push!(centers,y[draw_weights(rng,w)])
    end
    c=zeros(Int,length(y))
    for _ in 1:100
        old=copy(c)
        for i in eachindex(y); c[i]=argmin([sum(abs2,y[i]-x) for x in centers]); end
        for h in 1:k
            inds=findall(==(h),c)
            if isempty(inds)
                counts=[count(==(j),c) for j in 1:k]
                i=argmax([counts[c[j]]>1 ? sum(abs2,y[j]-centers[c[j]]) : -Inf for j in eachindex(y)])
                c[i]=h; inds=[i]
            end
            centers[h]=mean(view(y,inds))
        end
        c==old && break
    end
    c,centers
end
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
function run_chain(s,iterations,burnin,ex=Execution();storage_interval=25)
    0<=burnin<iterations || throw(ArgumentError("need 0 <= burnin < iterations"))
    storage_interval>=1 || throw(ArgumentError("storage interval must be positive"))
    names=scalar_names(); traces=zeros(Float64,iterations,length(names))
    partitions=Vector{Vector{Int}}(); saved_iterations=Int[]
    burn_seconds=0.0; retained_seconds=0.0; burn_bytes=0; retained_bytes=0; gcseconds=0.0; diagnostic_seconds=0.0
    t0=time_ns()
    for it in 1:iterations
        timed=@timed sweep!(s,ex)
        if it<=burnin
            burn_seconds+=timed.time; burn_bytes+=timed.bytes
        else
            retained_seconds+=timed.time; retained_bytes+=timed.bytes
        end
        gcseconds+=timed.gctime
        diagtime=@elapsed begin
            traces[it,:].=scalar_summary(s)
            if it%storage_interval==0
                push!(partitions,canonical_labels(core(s).c)); push!(saved_iterations,it)
            end
        end
        diagnostic_seconds+=diagtime
    end
    (names=names,traces=traces,partitions=partitions,saved_iterations=saved_iterations,
        burn_seconds=burn_seconds,retained_seconds=retained_seconds,burn_bytes=burn_bytes,retained_bytes=retained_bytes,
        gc_seconds=gcseconds,diagnostic_seconds=diagnostic_seconds,chain_wall_seconds=(time_ns()-t0)/1e9,
        counters=Dict(string(f)=>getfield(core(s).counters,f) for f in fieldnames(Counters)),
        final_c=canonical_labels(core(s).c),final_S=copy(core(s).S))
end
