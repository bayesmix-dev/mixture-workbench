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
