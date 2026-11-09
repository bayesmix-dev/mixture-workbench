struct Model{D,L}
    y::Vector{SVector{D,Float64}}
    truth::Vector{Int}
    lower::SVector{D,Float64}
    upper::SVector{D,Float64}
    invwidth::SVector{D,Float64}
    volume::Float64
    logvolume::Float64
    a::Float64
    tau::Float64
    omega::SMatrix{D,D,Float64,L}
    omegaL::SMatrix{D,D,Float64,L}
    logdetomega::Float64
    spectrum::Spectrum{D}
    seed::UInt64
end
function Model(y::Vector{SVector{D,Float64}},truth,lower,upper,spec; a=0.1,tau=D+2.0,omega=Matrix{Float64}(I,D,D),seed=UInt64(1)) where D
    lo=SVector{D,Float64}(lower); hi=SVector{D,Float64}(upper)
    all(hi.>lo) && tau>D-1 && a>0 || throw(ArgumentError("invalid model"))
    om=SMatrix{D,D,Float64}(omega); ol=SMatrix{D,D,Float64}(cholesky(Symmetric(om)).L)
    vol=prod(hi-lo)
    Model(y,Int.(truth),lo,hi,1.0./(hi-lo),vol,log(vol),Float64(a),Float64(tau),om,ol,2sum(log,diag(ol)),spec,UInt64(seed))
end
@inline inside(m::Model{D},x) where D=all(i->m.lower[i]<=x[i]<=m.upper[i],1:D)
@inline uniform_point(rng,m::Model{D}) where D=m.lower+(m.upper-m.lower).*SVector{D,Float64}(ntuple(_->rand(rng),D))
function features!(out,m::Model{D},S,x) where D
    u=(x-m.lower).*m.invwidth
    @inbounds for (j,mode) in enumerate(S)
        kind=m.spectrum.kind[mode]
        if kind==0
            out[j]=1.0
        else
            z=0.0
            for e in 1:D; z+=m.spectrum.frequencies[e,mode]*u[e]; end
            out[j]=sqrt(2.0)*(kind==1 ? cospi(2z) : sinpi(2z))
        end
    end
    out
end
function feature_matrix(m,S,xs)
    F=Matrix{Float64}(undef,length(S),length(xs))
    for h in eachindex(xs); features!(view(F,:,h),m,S,xs[h]); end
    F
end
function envelope(spec,S)
    n0=0; pairs=Set{Int}()
    for j in S
        spec.kind[j]==0 ? (n0+=1) : push!(pairs,spec.kind[j]==1 ? j : j-1)
    end
    Float64(n0+2length(pairs))  # dimensionless bound; divide by volume for intensity
end
Base.@kwdef mutable struct Counters
    births::Int=0
    deletions::Int=0
    loc_attempts::Int=0
    loc_accepts::Int=0
    loc_outside::Int=0
    spectral_added::Int=0
    spectral_removed::Int=0
    new_attempts::Int=0
    new_accepts::Int=0
    new_outside::Int=0
    new_rejects::Int=0
    palm_attempts::Int=0
    unused_occupied::Int=0
    near_rank::Int=0
    clamps::Int=0
    projector_rebuilds::Int=0
end
function orthobasis(F::Matrix{Float64},ct::Counters)
    M,k=size(F)
    k==0 && return zeros(M,0),0.0
    k<=M || return zeros(M,0),-Inf
    fac=qr(F)
    diagonal=abs.(diag(fac.R))
    if minimum(diagonal)<=1e-10*max(1.0,maximum(diagonal))
        ct.near_rank+=1
        sv=svd(F)
        minimum(sv.S)==0 && return sv.U,-Inf
        return sv.U,2sum(log,sv.S)
    end
    Q=fac.Q*Matrix{Float64}(I,M,k)
    Q,2sum(log,diagonal)
end
# Explicit residual norm instead of subtracting squared lengths; two passes.
function residual!(v,Q,f)
    copyto!(v,f)
    @inbounds for _ in 1:2, h in axes(Q,2)
        coeff=0.0
        for j in eachindex(v); coeff+=Q[j,h]*v[j]; end
        for j in eachindex(v); v[j]-=Q[j,h]*coeff; end
    end
    sum(abs2,v)
end
function bounded_ratio(r,b,ct::Counters)
    if r>b
        r-b <= 256eps(Float64)*max(1.0,b) || error("Residual exceeds envelope: $r > $b")
        ct.clamps+=1; return 1.0
    end
    r>=0 || error("negative squared norm")
    r/b
end
function sample_J(rng,F,ct= Counters(); Qcache=nothing)
    M,k=size(F)
    M==k && return collect(1:k)
    Q,ld=isnothing(Qcache) ? orthobasis(F,ct) : (copy(Qcache),0.0)
    isfinite(ld) || error("rank-deficient current occupied features")
    chosen=Int[]; weights=zeros(M); v=zeros(k)
    for s in k:-1:1
        for j in 1:M; weights[j]=sum(abs2,view(Q,j,1:s)); end
        for j in chosen; weights[j]=0.0; end
        row=draw_weights(rng,weights); push!(chosen,row)
        s==1 && break
        for h in 1:s; v[h]=Q[row,h]; end
        normv=norm(view(v,1:s))
        v[s]+=copysign(normv,v[s])
        scale=2/sum(abs2,view(v,1:s))
        # Right Householder application; last column discarded. No dense reflector.
        for j in 1:M
            coeff=0.0
            for h in 1:s; coeff+=Q[j,h]*v[h]; end
            coeff*=scale
            for h in 1:s-1; Q[j,h]-=coeff*v[h]; end
        end
        Q[row,1:s-1].=0.0
    end
    sort!(chosen)
end
function draw_weights(rng,w)
    total=sum(w); total>0 && isfinite(total) || error("invalid categorical weights")
    u=rand(rng)*total; accum=0.0
    for h in eachindex(w)
        accum+=w[h]
        u<accum && return h
    end
    findlast(>(0),w)::Int
end
function draw_logweights!(rng,w)
    mx=maximum(w); isfinite(mx) || error("all categorical log weights nonfinite")
    for j in eachindex(w); w[j]=exp(w[j]-mx); end
    draw_weights(rng,w)
end
function palm_sample(rng,m,S,occupied,ct=Counters();maxtries=10^7)
    F=feature_matrix(m,S,occupied); Q,ld=orthobasis(F,ct)
    isfinite(ld) || error("Palm conditioning has zero target")
    b=envelope(m.spectrum,S); M=length(S)
    out=copy(occupied); f=zeros(M); v=zeros(M)
    while length(out)<M
        found=false
        for _ in 1:maxtries
            ct.palm_attempts+=1
            x=uniform_point(rng,m); features!(f,m,S,x)
            r=residual!(v,Q,f)
            if rand(rng)<bounded_ratio(r,b,ct)
                r>0 || error("zero residual selected")
                push!(out,x); Q=hcat(Q,v/sqrt(r)); found=true; break
            end
        end
        found || error("Palm rejection retry limit exceeded; run aborted")
    end
    out
end
@inline function mvnormal_draw(rng,mu::SVector{D},L::SMatrix{D,D}) where D
    mu+L*SVector{D,Float64}(ntuple(_->randn(rng),D))
end
@inline function student_draw(rng,mu::SVector{D},L,df) where D
    z=SVector{D,Float64}(ntuple(_->randn(rng),D))
    mu+(L*z)/sqrt(rand(rng,Chisq(df))) # L L' = df * scale
end
@inline logmultigamma(a,d)=d*(d-1)/4*log(pi)+sum(loggamma(a+(1-j)/2) for j in 1:d)
@inline function student_logdensity(y,mu,L,df)  # L L' = df * scale
    d=length(y); z=LowerTriangular(L)\(y-mu)
    loggamma((df+d)/2)-loggamma(df/2)-d/2*log(pi)-sum(log,diag(L))-(df+d)/2*log1p(sum(abs2,z))
end
function iw_factor(rng,df,L::SMatrix{D,D,T,N}) where {D,T,N}
    # Bartlett A A' ~ Wishart(df,I), Sigma = L (A A')^-1 L'.
    A=zeros(MMatrix{D,D,Float64,N})
    for j in 1:D
        A[j,j]=sqrt(rand(rng,Chisq(df-j+1)))
        for i in j+1:D; A[i,j]=randn(rng); end
    end
    C=L/UpperTriangular(SMatrix(A)')
    SMatrix{D,D,Float64,N}(cholesky(Symmetric(C*C')).L)
end
mutable struct Cluster{D,L}
    n::Int
    mean::SVector{D,Float64}
    scatter::SMatrix{D,D,Float64,L}
    x::SVector{D,Float64}
    scaleL::SMatrix{D,D,Float64,L}
    logdet::Float64
    logpredconst::Float64
end
function Cluster(m::Model{D,L},x) where {D,L}
    Cluster(0,zero(x),zero(m.omega),x,m.omegaL,m.logdetomega,0.0)
end
function add!(c::Cluster,y)
    n=c.n; delta=y-c.mean
    c.mean+=delta/(n+1)
    c.scatter+=(n/(n+1))*(delta*delta')
    c.n+=1; c
end
function remove!(c::Cluster,y)
    c.n>0 || error("removing from empty cluster")
    if c.n==1
        c.n=0; c.mean=zero(c.mean); c.scatter=zero(c.scatter)
    else
        n=c.n; delta=y-c.mean
        c.mean-=delta/(n-1)
        c.scatter-=(n/(n-1))*(delta*delta'); c.n-=1
    end
    c
end
@inline function predictive_normalizer(m::Model{D},n) where D
    df=m.tau+n-D+1
    log(n+m.a)+loggamma((df+D)/2)-loggamma(df/2)-D/2*log(pi)
end
function refresh!(c::Cluster,m::Model{D},norms=nothing) where D
    delta=c.mean-c.x
    mat=m.omega+c.scatter+c.n*(delta*delta')
    c.scaleL=cholesky(Symmetric(mat)).L
    c.logdet=2sum(log,diag(c.scaleL))
    norm=isnothing(norms) ? predictive_normalizer(m,c.n) : norms[c.n+1]
    c.logpredconst=norm-c.logdet/2
    c
end
@inline function predictive_mass(c::Cluster,m::Model{D},y) where D
    z=LowerTriangular(c.scaleL)\(y-c.x)
    c.logpredconst-(m.tau+c.n+1)/2*log1p(sum(abs2,z))
end
function log_marginal(c::Cluster,m::Model{D}) where D
    n=c.n
    n==0 && return 0.0
    -n*D/2*log(pi)+m.tau/2*m.logdetomega-(m.tau+n)/2*c.logdet+
        logmultigamma((m.tau+n)/2,D)-logmultigamma(m.tau/2,D)
end
struct Execution
    mode::Symbol
    threads::Int
    function Execution(mode=:serial,threads=1;budget=Sys.CPU_THREADS)
        mode in (:serial,:chains,:within) || throw(ArgumentError("unknown parallel mode"))
        1<=threads<=min(Threads.nthreads(:default),budget) || throw(ArgumentError("requested concurrency exceeds pool/CPU budget"))
        mode!=:within && threads!=1 && throw(ArgumentError("internally serial mode requires threads=1"))
        new(mode,threads)
    end
end
function foreach_chunk(f,n,ex::Execution)
    if ex.threads==1
        f(1:n)
    else
        # Fixed item streams preserve draw order regardless of chunk assignment.
        nt=min(ex.threads,n)
        @sync for t in 1:nt
            r=fld((t-1)*n,nt)+1:fld(t*n,nt)
            Threads.@spawn f(r)
        end
    end
end
mutable struct Streams
    root::UInt64
    obs::Vector{Xoshiro}
    cov::Vector{Xoshiro}
    loc::Vector{Xoshiro}
    accept::Xoshiro
    spectral::Xoshiro
    palm::Xoshiro
    weights::Xoshiro
end
function Streams(seed,n)
    Streams(UInt64(seed),[Xoshiro(seedof(seed,1,i)) for i in 1:n],Xoshiro[],Xoshiro[],
        Xoshiro(seedof(seed,4)),Xoshiro(seedof(seed,5)),Xoshiro(seedof(seed,6)),Xoshiro(seedof(seed,7)))
end
function grow_streams!(s,n)
    for i in length(s.loc)+1:n
        push!(s.loc,Xoshiro(seedof(s.root,2,i))); push!(s.cov,Xoshiro(seedof(s.root,3,i)))
    end
end
