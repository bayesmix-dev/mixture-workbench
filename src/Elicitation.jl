module Elicitation
using GeneralDPPMixtures, Random, Statistics, LinearAlgebra
const G=GeneralDPPMixtures
"Section 6.2 of arXiv:2210.04140: cluster separation drop, with 5% pruning."
function radius(y;seed=221004140,kmax=10,minfraction=.05,starts=10)
    rng=Xoshiro(seed); ds=Float64[]; ks=Int[]; retained=Int[]; objectives=Float64[]
    for k in 2:kmax
        best=Inf; bestc=Int[]; bestx=typeof(y)()
        for _ in 1:starts
            c,x=G.kmeans_start(rng,y,k)
            score=sum(sum(abs2,y[i]-x[c[i]]) for i in eachindex(y))
            if score<best; best=score; bestc=c; bestx=x; end
        end
        keep=[h for h in 1:k if count(==(h),bestc)>=minfraction*length(y)]
        length(keep)>=2 || continue
        d=minimum(norm(bestx[h]-bestx[j]) for h in keep for j in keep if h<j)
        push!(ds,d); push!(ks,k); push!(retained,length(keep)); push!(objectives,best)
    end
    eligible=[i for i in 1:length(ks)-1 if ks[i+1]==ks[i]+1]
    isempty(eligible) && error("Radius rule has no consecutive eligible k values")
    j=eligible[argmax([ds[i]-ds[i+1] for i in eligible])]
    R=(ds[j]+ds[j+1])/2
    (radius=R,kstar=ks[j],ks=ks,dmin=ds,retained=retained,objectives=objectives,
     seed=seed,minfraction=minfraction,starts=starts,kmax=kmax)
end
end
