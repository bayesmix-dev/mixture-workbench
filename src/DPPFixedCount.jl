module DPPFixedCount

using GeneralDPPMixtures
const G=GeneralDPPMixtures

export sweep!, allocation!

function check_count(s,k)
    k isa Integer && k>0 && length(s.active)==k ||
        throw(ArgumentError("state must have the requested occupied count"))
    Int(k)
end

"""
One allocation update conditional on exactly k occupied clusters.

A singleton's allocation stays fixed: assigning it to any other existing group
would remove its occupied cluster. Its center can still change in the location
block. For a nonsingleton, all k existing centers stay present, their DPP factor
is constant, and the ordinary collapsed predictive masses are exact conditionals.
No new-component branch, birth, deletion, or point-process cache rebuild is used.
"""
function allocation!(s::G.MarginalState,i,k=length(s.active))
    target=check_count(s,k)
    m=s.model;old=s.c[i];y=m.y[i]
    s.clusters[old].n==1 && return old
    G.remove!(s.clusters[old],y)
    G.refresh!(s.clusters[old],m,s.predictive_norms)
    resize!(s.workw,target)
    for (j,h) in enumerate(s.active)
        s.workw[j]=G.predictive_mass(s.clusters[h],m,y)
    end
    h=s.active[G.draw_logweights!(s.streams.obs[i],s.workw)]
    G.add!(s.clusters[h],y)
    G.refresh!(s.clusters[h],m,s.predictive_norms)
    s.c[i]=h
    length(s.active)==target && all(id->s.clusters[id].n>0,s.active) ||
        error("fixed-count allocation changed occupied support")
    h
end

"""
Exact auxiliary kernel for the original DPP posterior conditional on occupied
count k. Locations and spectrum use the original marginal updates unchanged.
Use only on a fixed-count auxiliary; the full-target sampler must remain free to
change count. The count is checked before and after every complete sweep.
"""
function sweep!(s::G.MarginalState,k=length(s.active),ex=G.Execution())
    target=check_count(s,k)
    for i in eachindex(s.c);allocation!(s,i,target);end
    G.marginal_locations!(s,ex)
    G.spectral_update!(s)
    length(s.active)==target || error("fixed-count sweep changed occupied count")
    s
end

end
