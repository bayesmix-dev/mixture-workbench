module MFMBiasTempering

using GeneralDPPMixtures,Random
import ..MFMBaseline as MF
import ..MFMTarget as MT
import ..MFMLikelihoodTempering as LT
import ..DPPBiasTempering as BT
const G=GeneralDPPMixtures

export Workspace,CrossWorkspace,EntropyWorkspace,refresh!,check!,score,raw_score,raw_cross_score,raw_entropy,
       reference_metadata,native_allocation!,allocation!,sweep!,logtarget,swap_logratio

# Cache arithmetic is shared with DPP. Only allocation kernels and physical
# rollback differ; no DPP geometry/location law enters the NIW MFM target.
const Workspace=BT.Workspace
const CrossWorkspace=BT.CrossWorkspace
const refresh! = BT.refresh!
const check! = BT.check!
const score=BT.score
const raw_score=BT.raw_score
const raw_cross_score=BT.raw_cross_score
const reference_metadata=BT.reference_metadata
const swap_logratio=BT.swap_logratio

entropy_flog(n::Integer)=n==0 ? 0. : n*log(Float64(n))
function entropy_counts(c)
    !isempty(c) && all(h->h isa Integer && h>0,c) || throw(ArgumentError("entropy requires a nonempty positive-label partition"))
    counts=Dict{Int,Int}()
    for h in c;counts[h]=get(counts,h,0)+1;end
    counts
end
raw_entropy(c)=log(length(c))-sum(entropy_flog,values(entropy_counts(c)))/length(c)

"""Incremental partition entropy H(c), in nats, with no reference partition.

The mathematical range is [0,log(n)]. A single allocation changes only the
source/destination cluster counts. Counts are exact integers; the sum of
n_h*log(n_h) is floating point and is rebuilt once at the start of each sweep.
This is only a bounded target coordinate; no partition/count prior is changed.
"""
mutable struct EntropyWorkspace <: BT.AbstractWorkspace
    n::Int
    logn::Float64
    stateid::UInt
    allocations::Vector{Int}
    counts::Dict{Int,Int}
    sum_flog::Float64
end
function EntropyWorkspace(s::MF.State)
    counts=entropy_counts(s.c);n=length(s.c)
    EntropyWorkspace(n,log(n),objectid(s),copy(s.c),counts,sum(entropy_flog,values(counts)))
end
function BT.refresh!(w::EntropyWorkspace,s)
    length(s.c)==w.n || throw(ArgumentError("entropy state length differs"))
    counts=entropy_counts(s.c)
    w.stateid=objectid(s);w.allocations=copy(s.c);w.counts=counts
    w.sum_flog=sum(entropy_flog,values(counts));w
end
BT.score(w::EntropyWorkspace)=w.logn-w.sum_flog/w.n
BT.reference_metadata(w::EntropyWorkspace)=(score="log(n)-sum_h n_h*log(n_h)/n",units="nats",n=w.n,
    lower_bound=0.,upper_bound=w.logn,reference_free=true,cache="integer cluster counts; floating nlogn sum refreshed each sweep")
BT.depends_on(w::EntropyWorkspace,i)=true
function BT.proposal_change(w::EntropyWorkspace,i,old,new)
    old==new && return (delta=0.,data=0.)
    a=get(w.counts,old,0);b=get(w.counts,new,0);a>=1 || error("invalid entropy cluster-count cache")
    df=entropy_flog(a-1)-entropy_flog(a)+entropy_flog(b+1)-entropy_flog(b)
    (delta=-df/w.n,data=df)
end
function BT.accept_change!(w::EntropyWorkspace,i,old,new,change)
    if old!=new
        left=w.counts[old]-1
        left==0 ? delete!(w.counts,old) : (w.counts[old]=left)
        w.counts[new]=get(w.counts,new,0)+1
    end
    w.sum_flog+=change.data;w.allocations[i]=new;w
end

# Called after removing i. Occupied groups have their state-private NIW
# predictive. Unoccupied component labels share one prior predictive and
# their combined mass is (M-Kminus)*gamma. No count cap is introduced.
function allocation_scores!(scores,s::MF.State,i)
    k=length(s.active);canbirth=k<s.M;resize!(scores,k+Int(canbirth))
    for (j,h) in enumerate(s.active)
        scores[j]=log(s.clusters[h].n+s.gamma)+MF.logpred(s,h,s.model.y[i])
    end
    empty=canbirth ? MF.empty_slot!(s) : 0
    canbirth && (scores[end]=log(s.M-k)+log(s.gamma)+MF.logpred(s,empty,s.model.y[i]))
    empty
end

"""One original NIW Gibbs allocation at fixed M,gamma,Lambda.

This is the literal per-observation allocation stage of MF.sweep!, factored
without changing its arithmetic, destination-slot rule or RNG consumption.
The original MF.sweep! is never modified by this optional module.
"""
function native_allocation!(s::MF.State,i,scores=Float64[])
    i isa Integer && 1<=i<=length(s.c) || throw(ArgumentError("invalid observation index"))
    old=s.c[i];G.remove!(s.clusters[old],s.model.y[i]);MF.refresh!(s,old)
    s.clusters[old].n==0 && deleteat!(s.active,findfirst(==(old),s.active))
    k=length(s.active);empty=allocation_scores!(scores,s,i)
    j=G.draw_logweights!(s.rng,scores);h=j<=k ? s.active[j] : empty
    j>k && push!(s.active,h)
    s.c[i]=h;G.add!(s.clusters[h],s.model.y[i]);MF.refresh!(s,h)
    h
end

# c[i], the affected cluster records and predictive caches are physical state.
# Small O(Kmax) copies also restore any appended slot and active-list ordering.
# No model/data copy, native RNG, external MH RNG or counter rewind occurs.
snapshot(s,i)=(label=s.c[i],clusters=deepcopy(s.clusters),active=copy(s.active),
    predL=copy(s.predL),predmu=copy(s.predmu),prednorm=copy(s.prednorm))
function restore!(s,i,old)
    s.c[i]=old.label;s.clusters=old.clusters;s.active=old.active
    s.predL=old.predL;s.predmu=old.predmu;s.prednorm=old.prednorm;s
end

"""Exact one-i Metropolis correction of the native NIW heat bath.

For pi_bias(s)=pi_MFM(s)*exp(bias*F(c)), the reversible native Gibbs proposal
has correction bias*(Fnew-Fold). This includes singleton births/deaths and the
natural M>=K capacity. M,gamma,Lambda and private NIW parameters are fixed
within this transition. Rejection restores physical state/caches and retains
advanced native and external streams. At bias0 no extra MH draw is made.
"""
function allocation!(w::BT.AbstractWorkspace,s::MF.State,i,bias,rng;scores=Float64[])
    BT.check!(w,s)
    _allocation!(w,s,i,BT.checklambda(bias),rng,scores)
end
function _allocation!(w,s,i,bias,rng,scores)
    i isa Integer && 1<=i<=length(s.c) || throw(ArgumentError("invalid observation index"))
    objectid(s)==w.stateid && s.c[i]==w.allocations[i] || throw(ArgumentError("stale touched allocation"))
    old=s.c[i];oldk=length(s.active);oldt=BT.score(w)
    if bias==0.
        new=native_allocation!(s,i,scores);change=BT.proposal_change(w,i,old,new)
        BT.accept_change!(w,i,old,new,change)
        return (index=Int(i),old_label=old,proposed_label=new,accepted=true,oldK=oldk,proposedK=length(s.active),
            bias=bias,old_score=oldt,proposed_score=BT.score(w),score=BT.score(w),logratio=0.,logu=nothing,outcome=:native)
    end
    saved=snapshot(s,i)
    new=try
        native_allocation!(s,i,scores)
    catch
        restore!(s,i,saved);rethrow()
    end
    newk=length(s.active);change=BT.proposal_change(w,i,old,new)
    proposed=oldt+change.delta;ratio=bias*change.delta;logu=log(rand(rng));accepted=logu<min(0.,ratio)
    if accepted;BT.accept_change!(w,i,old,new,change);else;restore!(s,i,saved);end
    (index=Int(i),old_label=old,proposed_label=new,accepted=accepted,oldK=oldk,proposedK=newk,
        bias=bias,old_score=oldt,proposed_score=proposed,score=BT.score(w),logratio=ratio,logu=logu,
        outcome=accepted ? :accepted : :bias_rejected)
end

"""Compose individually biased allocations with unchanged hyperparameters.

The partition-only score is independent of M,gamma,Lambda given c. Therefore
the original exact unbounded M update, Gamma Lambda update and gamma MH update
remain invariant at every bias. Optional gamma slice updates stay external.
Cold bias0 calls the literal original whole MF.sweep!, then refreshes only
the separate bias cache. A systematic sweep receives no aggregate MH step.
An optional passive observer(state,event) runs after every nonzero-bias
allocation's commit or rollback. It must not mutate state or consume RNG.
The literal cold path ignores the observer and emits no allocation events.
"""
function sweep!(w::BT.AbstractWorkspace,s::MF.State,bias,rng;observer=nothing)
    b=BT.checklambda(bias)
    if b==0.
        MF.sweep!(s);BT.refresh!(w,s);return s
    end
    BT.refresh!(w,s);scores=Float64[]
    for i in eachindex(s.c)
        event=_allocation!(w,s,i,b,rng,scores)
        observer===nothing || observer(s,event)
    end
    LT.hyperparameters!(s)
    s
end
function logtarget(s::MF.State,w::BT.AbstractWorkspace,bias)
    BT.check!(w,s);MT.logtarget(s)+BT.checklambda(bias)*BT.score(w)
end

end
