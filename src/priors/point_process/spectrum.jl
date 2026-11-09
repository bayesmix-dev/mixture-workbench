"""Finite real Fourier spectrum; modes sorted by exact integer squared norm."""
struct Spectrum{D}
    frequencies::Matrix{Int}
    kind::Vector{UInt8}             # 0 constant, 1 cosine, 2 sine
    shell::Vector{Int}
    ranges::Vector{UnitRange{Int}}
    norms::Vector{Int}
    lambda::Vector{Float64}         # one per shell
    logodds::Vector{Float64}
    alpha::Float64
    nu::Float64
    c0::Float64
    cutoff::Float64
    retained_sum::Float64
    conditional_mean::Float64
    tail_estimate::Float64
    logp0::Float64
end
function lattice_representatives(::Val{D}, r2::Int) where D
    out = NTuple{D,Int}[]
    w = zeros(Int,D)
    function visit(e,left)
        if e>D
            j=findfirst(!iszero,w)
            (isnothing(j) || w[j]>0) && push!(out,Tuple(w))
            return
        end
        R=isqrt(left)
        for v in -R:R
            w[e]=v; visit(e+1,left-v*v)
        end
    end
    visit(1,r2)
    sort!(out;by=w->(sum(abs2,w),w))
end
function gaussian_lattice_sum(alpha,d,c0)
    R=ceil(Int,sqrt(-log(1e-20))/alpha)
    c0*(1+2sum(exp(-(alpha*j)^2) for j in 1:R))^d
end
function make_spectrum(d::Int; c0=0.99, target=10.0, nu=2.0, cutoff=1e-8)
    nu==2 || throw(ArgumentError("Primary implementation calibrates nu=2 only; sensitivity is not part of this comparison"))
    0<c0<1 && target>c0 && 0<cutoff<c0 || throw(ArgumentError("invalid spectrum parameters"))
    lo=0.01; hi=10.0
    for _ in 1:90
        mid=(lo+hi)/2
        if gaussian_lattice_sum(mid,d,c0)>target; lo=mid; else; hi=mid; end
    end
    alpha=(lo+hi)/2
    s=build_spectrum(Val(d),alpha,c0,nu,cutoff)
    s
end
function build_spectrum(::Val{D},alpha,c0,nu,cutoff) where D
    r2=floor(Int,log(c0/cutoff)^(2/nu)/alpha^2)
    reps=lattice_representatives(Val(D),r2)
    freq=NTuple{D,Int}[]; kinds=UInt8[]; ns=Int[]
    for w in reps
        norm2=sum(abs2,w)
        c0*exp(-(alpha*sqrt(norm2))^nu)>=cutoff || continue
        push!(freq,w); push!(kinds,norm2==0 ? 0 : 1); push!(ns,norm2)
        if norm2>0; push!(freq,w); push!(kinds,2); push!(ns,norm2); end
    end
    norms=unique(ns); ranges=UnitRange{Int}[]; shell=zeros(Int,length(ns)); lambdas=Float64[]
    for (h,r) in enumerate(norms)
        first=searchsortedfirst(ns,r); last=searchsortedlast(ns,r)
        push!(ranges,first:last); shell[first:last].=h
        push!(lambdas,c0*exp(-(alpha*sqrt(r))^nu))
    end
    mass=sum(length(r)*lambdas[i] for (i,r) in enumerate(ranges))
    lp0=sum(length(r)*log1p(-lambdas[i]) for (i,r) in enumerate(ranges))
    finer=lattice_representatives(Val(D),floor(Int,log(c0/1e-16)^(2/nu)/alpha^2))
    ref=sum((sum(abs2,w)==0 ? 1 : 2)*c0*exp(-(alpha*sqrt(sum(abs2,w)))^nu) for w in finer)
    Spectrum{D}(reduce(hcat,collect.(freq)),kinds,shell,ranges,norms,lambdas,
        log.(lambdas).-log1p.(-lambdas),alpha,nu,c0,cutoff,mass,mass/(-expm1(lp0)),max(0,ref-mass),lp0)
end
function shell_draw(rng,spec::Spectrum,forced::Vector{Int},logza::Float64)
    selected=copy(forced)
    for (h,range) in enumerate(spec.ranges)
        blocked=Int[j-first(range)+1 for j in forced if j in range]
        sort!(blocked)
        nf=length(range)-length(blocked)
        r=rand(rng,Binomial(nf,logistic(logza+spec.logodds[h])))
        for freepos in sparse_subset(rng,nf,r)
            pos=freepos
            for b in blocked; b<=pos ? (pos+=1) : break; end
            push!(selected,first(range)+pos-1)
        end
    end
    sort!(selected)
end
function prior_support(rng,spec; minimum=1)
    while true
        S=shell_draw(rng,spec,Int[],0.0)
        length(S)>=minimum && return S
    end
end
