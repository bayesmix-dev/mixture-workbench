abstract type PointPrior end
"Finite L ensemble from the EXACT eigenvalues used by the spectral samplers."
struct DPPPrior <: PointPrior
    modes::Vector{Int}
    roots::Vector{Float64}
end
DPPPrior(m)=DPPPrior(collect(eachindex(m.spectrum.kind)),
    [sqrt(exp(m.spectrum.logodds[h])) for h in m.spectrum.shell])
struct StraussPrior <: PointPrior
    beta::Float64               # activity times domain volume, fixed
    radius::Float64             # original marker units
    logalpha::Float64           # keep in log scale: exp(-n/20) may underflow
end
