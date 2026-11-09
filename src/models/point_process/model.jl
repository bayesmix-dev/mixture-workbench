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
