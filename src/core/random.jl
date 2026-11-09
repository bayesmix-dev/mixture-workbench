@inline function splitmix(x::UInt64)
    z=x+0x9e3779b97f4a7c15
    z=(z ⊻ (z>>30))*0xbf58476d1ce4e5b9
    z=(z ⊻ (z>>27))*0x94d049bb133111eb
    z ⊻ (z>>31)
end
"""Stable unsigned SplitMix64 fold of documented integer logical identifiers."""
function seedof(ids::Integer...)
    x=0x6a09e667f3bcc909
    for id in ids; x=splitmix(x ⊻ UInt64(id)); end
    x
end
@inline logadd(x,y)=max(x,y)+log1p(exp(-abs(x-y)))
@inline logistic(x)=x>=0 ? 1/(1+exp(-x)) : exp(x)/(1+exp(x))
function loggamma_draw(rng,a)
    a>0 || throw(ArgumentError("Gamma shape must be positive"))
    a<1 ? log(rand(rng,Gamma(a+1,1)))+log(rand(rng))/a : log(rand(rng,Gamma(a,1)))
end
function logbeta_draw(rng,a,b)
    x=loggamma_draw(rng,a); y=loggamma_draw(rng,b)
    x-logadd(x,y)
end
# Floyd sampling: O(number selected) storage/work; no full-shell shuffle.
function sparse_subset(rng,n,r)
    selected=Set{Int}()
    for j in n-r+1:n
        t=rand(rng,1:j)
        push!(selected,t in selected ? j : t)
    end
    sort!(collect(selected))
end
