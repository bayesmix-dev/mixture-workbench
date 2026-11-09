module DPPSplitMerge
# Fixed-B split–merge only. The old local transition is called without modification.
using GeneralDPPMixtures, LinearAlgebra, Random, StaticArrays, SpecialFunctions
const G = GeneralDPPMixtures
# Protect only MPFR's temporary precision changes. Every production BigFloat
# operation in this module is inside big_logdet; ordinary scans are state-local.
const BIGFLOAT_LOCK=ReentrantLock()
export Workspace, launch_partition, restricted_scan!, propose!, attempt!, sweep!, logA,
       cluster, location_law, physical_logdet, geometry_delta, raw_ratio, snapshot, audit

function cluster(m, members, x=zero(first(m.y)))
    c=G.Cluster(m,x)
    for i in sort(members); G.add!(c,m.y[i]); end
    G.refresh!(c,m); c
end
function logA(m::G.Model{D},c) where D
    c.n>0 || throw(ArgumentError("A requires a nonempty cluster"))
    r=c.n; L=cholesky(Symmetric(m.omega+c.scatter)).L
    -(r-1)*D/2*log(pi)-D/2*log(r)+m.tau/2*m.logdetomega-
        (m.tau+r-1)*sum(log,diag(L))+G.logmultigamma((m.tau+r-1)/2,D)-G.logmultigamma(m.tau/2,D)
end
function location_law(m::G.Model{D,L},c) where {D,L}
    c.mean,SMatrix{D,D,Float64,L}(cholesky(Symmetric((m.omega+c.scatter)/c.n)).L),m.tau+c.n-D
end

"""Pure, exactly five simultaneous Lloyd iterations, anchor-oriented, sorted-index sums."""
function launch_partition(data,D,i,j)
    i!=j && i in D && j in D || throw(ArgumentError("invalid anchors"))
    inds=sort(D); length(unique(inds))==length(inds) || error("duplicate union indices")
    labels=zeros(Int,length(data)); u=data[i]; v=data[j]
    for _ in 1:5
        for t in inds
            labels[t]=t==i ? 1 : t==j ? 2 : sum(abs2,data[t]-u)<=sum(abs2,data[t]-v) ? 1 : 2
        end
        su=zero(u); sv=zero(v); nu=0; nv=0
        for t in inds
            if labels[t]==1; su+=data[t]; nu+=1; else; sv+=data[t]; nv+=1; end
        end
        u=su/nu; v=sv/nv
    end
    labels
end
mutable struct Workspace{C}
    groups::Vector{C}
    labels::Vector{Int}
    scores::Vector{Float64}
end
Workspace(m)=Workspace([G.Cluster(m,zero(first(m.y))) for _ in 1:2],zeros(Int,length(m.y)),zeros(2))

"""One scan only. Forced and sampled modes share all probability arithmetic.
`forced` is an n-vector oriented by i→1,j→2, never a starting partition.
"""
function restricted_scan!(w,m,D,i,j,order,rng;forced=nothing,launch_data=m.y)
    sort(order)==sort(setdiff(D,[i,j])) || error("scan must visit each nonanchor exactly once")
    forced===nothing || (forced[i]==1 && forced[j]==2 && all(t->forced[t] in (1,2),D)) || error("invalid forced daughters")
    w.labels.=launch_partition(launch_data,D,i,j)
    for c in w.groups; c.n=0; c.mean=zero(c.mean); c.scatter=zero(c.scatter); end
    for t in sort(D); G.add!(w.groups[w.labels[t]],m.y[t]); end
    logs=0.0
    for t in order
        G.remove!(w.groups[w.labels[t]],m.y[t])
        for h in 1:2
            c=w.groups[h]; old=logA(m,c)
            # A temporary add/remove must not accumulate trial-roundoff in the groups.
            n,mu,sc=c.n,c.mean,c.scatter
            G.add!(c,m.y[t]); aug=logA(m,c)
            c.n=n; c.mean=mu; c.scatter=sc
            w.scores[h]=log(n+m.a)+aug-old
        end
        z=max(w.scores...); norm=z+log(exp(w.scores[1]-z)+exp(w.scores[2]-z))
        h=forced===nothing ? (rand(rng)<exp(w.scores[1]-norm) ? 1 : 2) : forced[t]
        logs+=w.scores[h]-norm
        w.labels[t]=h; G.add!(w.groups[h],m.y[t])
    end
    [sort([t for t in D if w.labels[t]==h]) for h in 1:2],logs
end

# Householder QR in ordinary cases; near rank loss is recomputed with the actual
# Fourier functions at two precisions. Ambiguous rank is an error, never jitter.
function big_logdet(m,S,xs,bits)
    lock(BIGFLOAT_LOCK) do
        setprecision(BigFloat,bits) do
            F=Matrix{BigFloat}(undef,length(S),length(xs))
            for (h,x) in enumerate(xs), (r,s) in enumerate(S)
                kind=m.spectrum.kind[s]
                z=sum(BigFloat(m.spectrum.frequencies[e,s])*(BigFloat(x[e])-BigFloat(m.lower[e]))/
                    (BigFloat(m.upper[e])-BigFloat(m.lower[e])) for e in eachindex(x))
                F[r,h]=kind==0 ? BigFloat(1) : sqrt(BigFloat(2))*(kind==1 ? cospi(2z) : sinpi(2z))
            end
            ds=abs.(diag(qr(F).R))
            any(iszero,ds) ? -Inf : Float64(2sum(log,ds))
        end
    end
end
function physical_logdet(m,S,xs)
    k=length(xs); k==0 && return 0.0; k>length(S) && return -Inf
    F=G.feature_matrix(m,S,xs)
    for a in 1:k, b in 1:a-1
        xs[a]==xs[b] && return -Inf
    end
    ds=abs.(diag(qr(F).R))
    if minimum(ds)>1e-10*max(1.,maximum(ds))
        return 2sum(log,ds)-k*m.logvolume
    end
    a=big_logdet(m,S,xs,256); b=big_logdet(m,S,xs,512)
    a==b==-Inf && return -Inf
    isfinite(a) && isfinite(b) && abs(a-b)<1e-7 || error("unresolved numerical rank loss in DPP geometry")
    b-k*m.logvolume
end
function geometry_delta(m,S,unaffected,oldxs,newxs)
    # Only unaffected locations are projected out. Physical volume is retained.
    allold=vcat(unaffected,oldxs); allnew=vcat(unaffected,newxs)
    for a in eachindex(allnew), b in 1:a-1
        allnew[a]==allnew[b] && return -Inf
    end
    F0=G.feature_matrix(m,S,unaffected)
    Q,ld0=G.orthobasis(F0,G.Counters())
    isfinite(ld0) || error("invalid unaffected configuration")
    function residual_ld(xs)
        Qr=copy(Q); total=0.0; f=zeros(length(S)); v=similar(f)
        for x in xs
            G.features!(f,m,S,x); r=G.residual!(v,Qr,f)
            r>1e-20*max(1.,sum(abs2,f)) || return nothing
            total+=log(r); Qr=hcat(Qr,v/sqrt(r))
        end
        total
    end
    a=residual_ld(oldxs); b=residual_ld(newxs)
    if a===nothing || b===nothing || size(F0,2)>0 && minimum(svdvals(F0))<1e-10
        old=physical_logdet(m,S,allold); isfinite(old) || error("current state has zero DPP target")
        return physical_logdet(m,S,allnew)-old
    end
    b-a-(length(newxs)-length(oldxs))*m.logvolume
end

mutable struct Proposal{X}
    kind::Symbol
    anchors::Tuple{Int,Int}
    order::Vector{Int}
    oldids::Vector{Int}
    oldgroups::Vector{Vector{Int}}
    newgroups::Vector{Vector{Int}}
    oldxs::Vector{X}
    newxs::Vector{X}
    k::Int
    M::Int
    Gterm::Union{Nothing,Float64}
    H::Union{Nothing,Float64}
    DK::Union{Nothing,Float64}
    logs::Union{Nothing,Float64}
    ratio::Union{Nothing,Float64}
    logu::Union{Nothing,Float64}
    outcome::Symbol
end
function propose!(w,s,rng;anchors=nothing,order=nothing)
    m=s.model; n=length(s.c); n<2 && return nothing
    i,j=if anchors===nothing
        a=rand(rng,1:n); b=rand(rng,1:n-1); (a,b>=a ? b+1 : b)
    else
        anchors
    end
    1<=i<=n && 1<=j<=n && i!=j || error("invalid anchors")
    ids=unique([s.c[i],s.c[j]]); split=length(ids)==1
    old=[findall(==(h),s.c) for h in ids]; D=sort(vcat(old...))
    p=Proposal(split ? :split : :merge,(i,j),Int[],ids,old,Vector{Int}[],
        [s.clusters[h].x for h in ids],typeof(first(m.y))[],length(s.active),length(s.S),
        nothing,nothing,nothing,nothing,nothing,nothing,:constructed)
    if split && p.k==p.M; p.outcome=:capacity; return p; end
    p.order=order===nothing ? shuffle(rng,setdiff(D,[i,j])) : copy(order)
    forced=if split
        nothing
    else
        f=zeros(Int,n); f[old[1]].=1; f[old[2]].=2; f
    end
    daughters,logs=restricted_scan!(w,m,D,i,j,p.order,rng;forced)
    p.logs=logs
    c1=cluster(m,daughters[1]); c2=cluster(m,daughters[2]); parent=cluster(m,D)
    p.Gterm=loggamma(m.a+c1.n)+loggamma(m.a+c2.n)-loggamma(m.a+parent.n)-loggamma(m.a)
    p.H=logA(m,c1)+logA(m,c2)-logA(m,parent)
    p.newgroups=split ? daughters : [D]
    for c in (split ? (c1,c2) : (parent,))
        mu,L,df=location_law(m,c); push!(p.newxs,G.student_draw(rng,mu,L,df))
    end
    if !all(x->G.inside(m,x),p.newxs); p.outcome=:outside_domain; return p; end
    unaffected=[s.clusters[h].x for h in s.active if !(h in ids)]
    p.DK=geometry_delta(m,s.S,unaffected,p.oldxs,p.newxs)
    if p.DK==-Inf; p.outcome=:zero_determinant; return p; end
    p.ratio=split ? p.Gterm+p.H+p.DK-logs : -p.Gterm-p.H+p.DK+logs
    isfinite(p.ratio) || error("nonfinite SM acceptance ratio")
    p.logu=log(rand(rng)); p.outcome=p.logu<min(0.,p.ratio) ? :accepted : :mh_rejected
    p
end

function commit!(s,p)
    p.outcome==:accepted || return s
    m=s.model; ids=copy(p.oldids)
    if p.kind==:split
        h=findfirst(t->s.clusters[t].n==0,eachindex(s.clusters))
        push!(ids,h===nothing ? length(s.clusters)+1 : h)
    end
    cl=copy(s.clusters); c=copy(s.c)
    for (j,members) in enumerate(p.newgroups)
        h=ids[j]; cc=cluster(m,members,p.newxs[j])
        h>length(cl) ? push!(cl,cc) : (cl[h]=cc)
        c[members].=h
    end
    if p.kind==:merge; cl[ids[2]]=G.Cluster(m,zero(first(m.y))); end
    active=[h for h in s.active if !(h in p.oldids)]
    append!(active,ids[1:length(p.newgroups)])
    # Storage orientation is irrelevant; memberships and locations move together.
    sort!(active;by=h->findfirst(==(h),c))
    F=G.feature_matrix(m,s.S,[cl[h].x for h in active]); ct=G.Counters()
    Q,ld=G.orthobasis(F,ct)
    physical=physical_logdet(m,s.S,[cl[h].x for h in active])
    isfinite(ld) && isfinite(physical) && abs(ld-length(active)*m.logvolume-physical)<1e-7 || error("accepted-state feature cache inconsistency")
    # All fallible numerical work precedes this commit. B is never assigned.
    G.grow_streams!(s.streams,length(cl))
    s.clusters=cl; s.c=c; s.active=active; s.F=F; s.Q=Q; s.ld=ld
    s.counters.projector_rebuilds+=1
    s
end
function attempt!(w,s,rng;observer=nothing,kwargs...)
    p=propose!(w,s,rng;kwargs...)
    p===nothing && return nothing
    observer===nothing || observer(s,p)
    commit!(s,p); p
end
function sweep!(s,w,rng;sm_attempts=1,observer=nothing)
    sm_attempts in (0,1) || throw(ArgumentError("sm_attempts must be 0 or 1"))
    G.marginal_collapsed!(s,G.Execution())
    sm_attempts==1 && attempt!(w,s,rng;observer)
    s
end

function snapshot(s)
    Dict("canonical_labels"=>G.canonical_labels(s.c),"stable_ids"=>copy(s.c),"selected_modes"=>copy(s.S),
        "clusters"=>[Dict("id"=>h,"members"=>findall(==(h),s.c),"x"=>collect(s.clusters[h].x)) for h in s.active])
end
function raw_ratio(m,S,unaffected,oldgroups,oldxs,newgroups,newxs,logs,kind)
    # Independent unsimplified density route: full physical Gram matrix and log m.
    function target(groups,xs)
        allxs=vcat(unaffected,xs); F=G.feature_matrix(m,S,allxs)/sqrt(m.volume)
        all(x->G.inside(m,x),allxs) || return -Inf
        K=Symmetric(F'F); ld=2sum(log,diag(cholesky(K).L))
        ld+sum(loggamma(m.a+length(g))-loggamma(m.a)+G.log_marginal(cluster(m,g,x),m) for (g,x) in zip(groups,xs))
    end
    logf(groups,xs)=sum(begin
        c=cluster(m,g); mu,L,df=location_law(m,c); G.student_logdensity(x,mu,L,df)
    end for (g,x) in zip(groups,xs))
    delta=target(newgroups,newxs)-target(oldgroups,oldxs)
    qforward=logf(newgroups,newxs)+(kind==:split ? logs : 0.)
    qreverse=logf(oldgroups,oldxs)+(kind==:merge ? logs : 0.)
    (target_difference=delta,log_forward=qforward,log_reverse=qreverse,raw=delta+qreverse-qforward)
end
function audit(s)
    m=s.model; errmean=0.; errscatter=0.; errfactor=0.; errcount=0
    canonical=0.; cached=0.; likelihood=0.; partition=0.
    for h in s.active
        old=s.clusters[h]; c=cluster(m,findall(==(h),s.c),old.x)
        errmean=max(errmean,maximum(abs,c.mean-old.mean)); errscatter=max(errscatter,maximum(abs,c.scatter-old.scatter))
        errfactor=max(errfactor,maximum(abs,c.scaleL-old.scaleL)); errcount=max(errcount,abs(c.n-old.n))
        part=loggamma(m.a+c.n)-loggamma(m.a)
        canonical+=part+logA(m,c)-m.logvolume; cached+=part+logA(m,old)-m.logvolume
        partition+=part; likelihood+=G.log_marginal(c,m)
    end
    M=length(s.S); spectral=sum(m.spectrum.logodds[m.spectrum.shell[j]] for j in s.S)
    countterm=loggamma(m.a*M)-loggamma(length(s.c)+m.a*M)
    ld=physical_logdet(m,s.S,[s.clusters[h].x for h in s.active])
    F=G.feature_matrix(m,s.S,[s.clusters[h].x for h in s.active])
    errors=Dict("count"=>errcount,"mean_max_abs"=>errmean,"scatter_max_abs"=>errscatter,"factor_max_abs"=>errfactor,
        "feature_max_abs"=>maximum(abs,F-s.F),"logdet_abs"=>abs(ld-(s.ld-length(s.active)*m.logvolume)),
        "cached_minus_deterministic_unboxed_score"=>cached-canonical)
    membership_ok=sort(unique(s.c))==sort(s.active) && sum(s.clusters[h].n for h in s.active)==length(s.c)
    status=membership_ok && errcount==0 && errmean<1e-7 && errscatter<1e-6 && errfactor<1e-6 && errors["feature_max_abs"]<1e-12 && errors["logdet_abs"]<1e-7 ? "passed" : "cache_discrepancy"
    Dict("status"=>status,"errors"=>errors,"canonical_labels"=>G.canonical_labels(s.c),
        "deterministic_unboxed_score"=>canonical,"unboxed_score_definition"=>"sum log rising factorial(a,n_C) + log A_C - log volume; no count or determinant term",
        "joint_target_terms"=>Dict("spectral_logodds"=>spectral,
        "count_gamma"=>countterm,"physical_logdet"=>ld,"partition_gamma"=>partition,"log_m_likelihood"=>likelihood),
        "joint_log_target_omitting_constant_B_baseline"=>spectral+countterm+ld+partition+likelihood)
end
end
