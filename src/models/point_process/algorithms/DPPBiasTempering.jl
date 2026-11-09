module DPPBiasTempering

using GeneralDPPMixtures, Random, SHA
const G=GeneralDPPMixtures

export AbstractWorkspace, Workspace, CrossWorkspace, refresh!, check!, score, raw_score, raw_cross_score, variation_information,
       proposal_change, accept_change!, depends_on,
       allocation!, sweep!, canonicalize_occupied_clusters!, logtarget, swap_logratio, reference_metadata

flog(n::Integer)=n==0 ? 0. : n*log(Float64(n))
function labels(c,n)
    length(c)==n && all(h->h isa Integer && h>0,c) || throw(ArgumentError("reference must contain n positive integer labels"))
    G.canonical_labels(Int.(collect(c)))
end
function contingency(c,r)
    cells=Dict{Tuple{Int,Int},Int}()
    for i in eachindex(c)
        key=(c[i],r[i]);cells[key]=get(cells,key,0)+1
    end
    cells
end
function entropy(c)
    n=length(c);counts=Dict{Int,Int}()
    for h in c;counts[h]=get(counts,h,0)+1;end
    log(n)-sum(flog,values(counts))/n
end
"""Variation of information in nats; both arguments describe the same observations."""
function variation_information(c,r)
    n=length(c);n>0 || throw(ArgumentError("empty partition"))
    a=labels(c,n);b=labels(r,n)
    2*(log(n)-sum(flog,values(contingency(a,b)))/n)-entropy(a)-entropy(b)
end
raw_score(c,a,b)=variation_information(c,a)-variation_information(c,b)

"""Frozen two-reference VI contrast and state-specific incremental cache.

T(c)=VI(c,reference_a)-VI(c,reference_b). Positive bias favors reference_b.
References are copied, canonicalized and never adapted. The VI metric gives
abs(T(c)) <= VI(reference_a,reference_b), so every finite bias is proper.
Use refresh! after any external partition-changing move, or use sweep!, which
refreshes once before its allocation scan. Public allocation! checks the whole
cached partition by default; the internal sweep needs only a constant-time
touched-observation check after its initial refresh.
An external partition kernel at nonzero bias needs its own lambda*DeltaT
correction; refreshing its cache alone does not correct its stationary target.
"""
abstract type AbstractWorkspace end
mutable struct Workspace <: AbstractWorkspace
    reference_a::Vector{Int}
    reference_b::Vector{Int}
    reference_constant::Float64
    reference_distance::Float64
    reference_hashes::Tuple{String,String}
    stateid::UInt
    allocations::Vector{Int}
    cells_a::Dict{Tuple{Int,Int},Int}
    cells_b::Dict{Tuple{Int,Int},Int}
    sum_a::Float64
    sum_b::Float64
end
function Workspace(s,reference_a,reference_b)
    n=length(s.c);n>0 || throw(ArgumentError("empty state"))
    a=labels(reference_a,n);b=labels(reference_b,n)
    hashes=map(c->bytes2hex(sha256(reinterpret(UInt8,Int64.(c)))),(a,b))
    w=Workspace(a,b,-entropy(a)+entropy(b),variation_information(a,b),hashes,
        objectid(s),Int[],Dict{Tuple{Int,Int},Int}(),Dict{Tuple{Int,Int},Int}(),0.,0.)
    refresh!(w,s)
end
function refresh!(w::Workspace,s)
    length(s.c)==length(w.reference_a) || throw(ArgumentError("state/reference lengths differ"))
    all(>(0),s.c) || throw(ArgumentError("invalid current labels"))
    w.stateid=objectid(s);w.allocations=copy(s.c)
    w.cells_a=contingency(s.c,w.reference_a);w.cells_b=contingency(s.c,w.reference_b)
    w.sum_a=sum(flog,values(w.cells_a));w.sum_b=sum(flog,values(w.cells_b));w
end
function check!(w::AbstractWorkspace,s)
    objectid(s)==w.stateid && s.c==w.allocations || throw(ArgumentError("stale bias cache; call refresh! after external partition changes"))
    w
end
score(w::Workspace)=2*(w.sum_b-w.sum_a)/length(w.allocations)+w.reference_constant
function reference_metadata(w::Workspace)
    (reference_a_sha256=w.reference_hashes[1],reference_b_sha256=w.reference_hashes[2],
     reference_vi=w.reference_distance,score="VI(c,reference_a)-VI(c,reference_b)",units="nats")
end
function delta_sum(cells,ref,i,old,new)
    old==new && return 0.
    a=get(cells,(old,ref[i]),0);b=get(cells,(new,ref[i]),0)
    a>=1 || error("invalid bias contingency cache")
    flog(a-1)-flog(a)+flog(b+1)-flog(b)
end
function change_cells!(cells,ref,i,old,new)
    old==new && return
    a=(old,ref[i]);b=(new,ref[i]);left=cells[a]-1
    left==0 ? delete!(cells,a) : (cells[a]=left)
    cells[b]=get(cells,b,0)+1
end
function change!(w,i,old,new,da,db)
    change_cells!(w.cells_a,w.reference_a,i,old,new)
    change_cells!(w.cells_b,w.reference_b,i,old,new)
    w.sum_a+=da;w.sum_b+=db;w.allocations[i]=new;w
end
function proposal_change(w::Workspace,i,old,new)
    da=delta_sum(w.cells_a,w.reference_a,i,old,new);db=delta_sum(w.cells_b,w.reference_b,i,old,new)
    (delta=2*(db-da)/length(w.allocations),data=(da,db))
end
accept_change!(w::Workspace,i,old,new,change)=change!(w,i,old,new,change.data...)
depends_on(w::Workspace,i)=true

"""Frozen cross-set coassignment fraction with an exact integer numerator.

The two copied nonempty index sets must be disjoint. T(c) is the fraction of
the fixed A-by-B pairs assigned to the same cluster. Single-point increments
use the opposite set's cluster counts and take constant time. This object is
only a target coordinate/cache; it does not adapt either frozen set.
"""
mutable struct CrossWorkspace <: AbstractWorkspace
    a::Vector{Int}
    b::Vector{Int}
    in_a::BitVector
    in_b::BitVector
    denominator::Int
    index_hashes::Tuple{String,String}
    stateid::UInt
    allocations::Vector{Int}
    counts_a::Dict{Int,Int}
    counts_b::Dict{Int,Int}
    cross_count::Int
end
function indices(values,n)
    all(i->i isa Integer && 1<=i<=n,values) || throw(ArgumentError("fixed sets require valid integer indices"))
    out=sort(Int.(collect(values)))
    !isempty(out) && length(unique(out))==length(out) || throw(ArgumentError("fixed sets must be nonempty with no duplicate indices"))
    out
end
function CrossWorkspace(s,A,B)
    n=length(s.c);a=indices(A,n);b=indices(B,n)
    isempty(intersect(a,b)) || throw(ArgumentError("cross-set coordinate requires disjoint fixed sets"))
    ia=falses(n);ib=falses(n);ia[a].=true;ib[b].=true
    hashes=map(c->bytes2hex(sha256(reinterpret(UInt8,Int64.(c)))),(a,b))
    w=CrossWorkspace(a,b,ia,ib,Base.checked_mul(length(a),length(b)),hashes,objectid(s),Int[],Dict{Int,Int}(),Dict{Int,Int}(),0)
    refresh!(w,s)
end
function set_counts(c,ids)
    out=Dict{Int,Int}()
    for i in ids;h=c[i];out[h]=get(out,h,0)+1;end
    out
end
function raw_cross_score(c,A,B)
    a=indices(A,length(c));b=indices(B,length(c))
    isempty(intersect(a,b)) || throw(ArgumentError("cross sets overlap"))
    ca=set_counts(c,a);cb=set_counts(c,b)
    sum(n*get(cb,h,0) for (h,n) in ca)/Base.checked_mul(length(a),length(b))
end
function refresh!(w::CrossWorkspace,s)
    length(s.c)==length(w.in_a) && all(>(0),s.c) || throw(ArgumentError("state/reference lengths or labels differ"))
    w.stateid=objectid(s);w.allocations=copy(s.c)
    w.counts_a=set_counts(s.c,w.a);w.counts_b=set_counts(s.c,w.b)
    w.cross_count=sum(n*get(w.counts_b,h,0) for (h,n) in w.counts_a);w
end
score(w::CrossWorkspace)=w.cross_count/w.denominator
depends_on(w::CrossWorkspace,i)=w.in_a[i] || w.in_b[i]
function reference_metadata(w::CrossWorkspace)
    (set_a_sha256=w.index_hashes[1],set_b_sha256=w.index_hashes[2],set_a_size=length(w.a),set_b_size=length(w.b),
     denominator=w.denominator,score="sum_h n_Ah*n_Bh/(length(A)*length(B))",units="fraction")
end
function proposal_change(w::CrossWorkspace,i,old,new)
    counts=w.in_a[i] ? w.counts_b : w.in_b[i] ? w.counts_a : nothing
    delta=counts===nothing || old==new ? 0 : get(counts,new,0)-get(counts,old,0)
    (delta=delta/w.denominator,data=delta)
end
function accept_change!(w::CrossWorkspace,i,old,new,change)
    counts=w.in_a[i] ? w.counts_a : w.in_b[i] ? w.counts_b : nothing
    if old!=new && counts!==nothing
        remaining=counts[old]-1
        remaining==0 ? delete!(counts,old) : (counts[old]=remaining)
        counts[new]=get(counts,new,0)+1
    end
    w.cross_count+=change.data;w.allocations[i]=new
    0<=w.cross_count<=w.denominator || error("cross coassignment numerator outside its exact range")
    w
end
function checklambda(lambda)
    lambda isa Real && isfinite(lambda) || throw(ArgumentError("bias must be a finite real number"))
    value=Float64(lambda)
    isfinite(value) || throw(ArgumentError("bias must have a finite Float64 representation"))
    value
end

# Native allocation changes one label and at most two cluster records. Keeping
# all small cluster records avoids finding its destination before proposing.
# Geometry rebuilds replace F/Q, so their old arrays can be retained directly.
# Random streams and counters are deliberately NOT part of this rollback.
function snapshot(s,i)
    (label=s.c[i],clusters=deepcopy(s.clusters),active=copy(s.active),F=s.F,Q=s.Q,
     ld=s.ld,bound=s.bound,workf=copy(s.workf),workv=copy(s.workv),workw=copy(s.workw))
end
function restore!(s,i,old)
    s.c[i]=old.label;s.clusters=old.clusters;s.active=old.active
    s.F=old.F;s.Q=old.Q;s.ld=old.ld;s.bound=old.bound
    s.workf=old.workf;s.workv=old.workv;s.workw=old.workw;s
end

"""One exact biased allocation transition; no covariance/likelihood power changes.

The native single-observation allocation is a reversible heat bath on the cold
unlabelled occupied-partition/center state, including singleton birth/death.
Its MH correction for pi(s)*exp(lambda*T(c)) is therefore lambda*DeltaT.
Rejection restores all physical/cache fields while retaining advanced native
and external MH streams and proposal counters. Scratch is also restored.
At lambda=0 the native primitive is called directly, with no extra RNG draw.
For a CrossWorkspace, observations outside both fixed sets have zero DeltaT.
At nonzero bias these still retain the transactional snapshot and MH draw,
including the same physical rollback if the native primitive aborts.
"""
function allocation!(w::AbstractWorkspace,s::G.MarginalState,i,lambda,rng;kwargs...)
    check!(w,s)
    _allocation!(w,s,i,checklambda(lambda),rng;kwargs...)
end
function _allocation!(w,s,i,lambda,rng;kwargs...)
    i isa Integer && 1<=i<=length(s.c) || throw(ArgumentError("invalid observation index"))
    objectid(s)==w.stateid && s.c[i]==w.allocations[i] || throw(ArgumentError("stale touched allocation"))
    old=s.c[i];oldk=length(s.active);oldt=score(w)
    if lambda==0.
        new=G.allocation!(s,i;kwargs...)
        change=proposal_change(w,i,old,new);accept_change!(w,i,old,new,change)
        return (index=Int(i),old_label=old,proposed_label=new,accepted=true,oldK=oldk,proposedK=length(s.active),
            lambda=lambda,old_score=oldt,proposed_score=score(w),score=score(w),logratio=0.,logu=nothing,outcome=:native)
    end
    saved=snapshot(s,i)
    new=try
        G.allocation!(s,i;kwargs...)
    catch
        restore!(s,i,saved);rethrow()
    end
    newk=length(s.active);change=proposal_change(w,i,old,new)
    proposed=oldt+change.delta;ratio=lambda*change.delta;logu=log(rand(rng));accepted=logu<min(0.,ratio)
    if accepted;accept_change!(w,i,old,new,change);else;restore!(s,i,saved);end
    (index=Int(i),old_label=old,proposed_label=new,accepted=accepted,oldK=oldk,proposedK=newk,
     lambda=lambda,old_score=oldt,proposed_score=proposed,score=score(w),logratio=ratio,logu=logu,
     outcome=accepted ? :accepted : :bias_rejected)
end

"""Rebuild occupied sufficient statistics in ascending observation order.

Only derived Cluster records are replaced. Labels, occupied order, current
locations, inactive records, geometry, predictive-normalizer table, counters
and all random streams remain unchanged. Counts/membership and every fresh
factor are checked before the cluster-vector assignment, so failure is atomic.
The label-based bias cache remains valid without refresh or invalidation.
"""
function canonicalize_occupied_clusters!(s::G.MarginalState)
    m=s.model;n=length(s.c);old=s.clusters;occupied=falses(length(old))
    n==length(m.y) && n>0 && length(s.predictive_norms)>=n+1 || throw(ArgumentError("invalid state/data/predictive-normalizer dimensions"))
    for h in s.active
        1<=h<=length(old) && !occupied[h] && old[h].n>0 || throw(ArgumentError("invalid occupied cluster IDs/counts"))
        occupied[h]=true
    end
    all(h->occupied[h] || old[h].n==0,eachindex(old)) || throw(ArgumentError("inactive cluster has observations"))
    fresh=copy(old)
    for h in s.active;fresh[h]=G.Cluster(m,old[h].x);end
    for i in eachindex(s.c)
        h=s.c[i]
        1<=h<=length(old) && occupied[h] || throw(ArgumentError("allocation is absent from occupied clusters"))
        G.add!(fresh[h],m.y[i])
    end
    for h in s.active
        fresh[h].n==old[h].n && fresh[h].n>0 || throw(ArgumentError("cached cluster count differs from allocations"))
        G.refresh!(fresh[h],m,s.predictive_norms)
        c=fresh[h]
        all(isfinite,c.mean) && all(isfinite,c.scatter) && all(isfinite,c.scaleL) &&
            isfinite(c.logdet) && isfinite(c.logpredconst) || error("nonfinite canonical cluster factor")
    end
    s.clusters=fresh
    s
end

"""Correct allocations individually; locations and spectrum leave T unchanged.

A full systematic cold sweep is not generally reversible and must not receive
one aggregate bias-only MH correction. Here each single-i corrected kernel is
invariant; their systematic composition is invariant as well. At zero bias the
original whole sweep is called byte-for-byte, then only the separate cache is
refreshed. For nonzero bias, a single O(n) refresh precedes O(1) contrast updates.
After the allocation scan, occupied sufficient statistics are rebuilt in
ascending observation order before location/spectral updates. This prevents
roundoff from accumulating across sweeps while preserving the same labels and
locations and leaving the literal zero-bias path unchanged.
The optional observer(state,event) runs after each nonzero-bias allocation has
committed or rolled back. It must be passive; no per-observation data are stored
by this module. At zero bias the observer is ignored and callers may inspect
the native aggregate counters instead. Allocation keyword overrides apply only
to nonzero bias; a zero-bias sweep with such overrides is rejected rather than
silently changing or bypassing the original whole-sweep call.
"""
function sweep!(w::AbstractWorkspace,s::G.MarginalState,lambda,rng,ex=G.Execution();observer=nothing,kwargs...)
    bias=checklambda(lambda)
    if bias==0.
        isempty(kwargs) || throw(ArgumentError("zero-bias whole sweeps do not accept allocation keyword overrides"))
        G.marginal_collapsed!(s,ex);refresh!(w,s);return s
    end
    refresh!(w,s)
    for i in eachindex(s.c)
        event=_allocation!(w,s,i,bias,rng;kwargs...)
        observer===nothing || observer(s,event)
    end
    canonicalize_occupied_clusters!(s)
    G.marginal_locations!(s,ex);G.spectral_update!(s)
    # A spectral rebuild may grow scratch with uninitialized tails. These
    # nondensity buffers are overwritten before use; canonicalizing them keeps
    # checkpoints independent of allocations made by a passive observer.
    fill!(s.workf,0.);fill!(s.workv,0.);fill!(s.workw,0.);s
end
function logtarget(s::G.MarginalState,w::AbstractWorkspace,lambda)
    check!(w,s);G.collapsed_logtarget(s)+checklambda(lambda)*score(w)
end
function swap_logratio(score_a::Real,score_b::Real,lambda_a,lambda_b)
    isfinite(score_a) && isfinite(score_b) || throw(ArgumentError("nonfinite contrast"))
    (checklambda(lambda_a)-checklambda(lambda_b))*(score_b-score_a)
end

end
