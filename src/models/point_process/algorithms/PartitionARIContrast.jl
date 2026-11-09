module PartitionARIContrast

using GeneralDPPMixtures,SHA
import ..DPPBiasTempering as BT
const G=GeneralDPPMixtures

export Workspace,score,raw_score,raw_ari,refresh!,check!,proposal_change,
       accept_change!,depends_on,reference_metadata
const score=BT.score
const refresh! = BT.refresh!
const check! = BT.check!
const proposal_change=BT.proposal_change
const accept_change! = BT.accept_change!
const depends_on=BT.depends_on
const reference_metadata=BT.reference_metadata

# Copy-owned reference storage has no public mutation API. As elsewhere in
# Julia, getfield can bypass an abstraction; callers must not mutate internals.
struct FrozenLabels <: AbstractVector{Int}
    values::Vector{Int}
end
Base.size(r::FrozenLabels)=size(r.values)
Base.getindex(r::FrozenLabels,i::Int)=r.values[i]
Base.IndexStyle(::Type{FrozenLabels})=IndexLinear()

checked_add(a::Int128,b::Int128)=Base.Checked.checked_add(a,b)
checked_sub(a::Int128,b::Int128)=Base.Checked.checked_sub(a,b)
checked_mul(a::Int128,b::Int128)=Base.Checked.checked_mul(a,b)
function paircount(n::Integer)
    n>=0 || throw(ArgumentError("negative cluster count"))
    x=Int128(n);checked_mul(x,x-1)÷2
end
function valid_labels(c,n)
    length(c)==n && all(h->h isa Integer && h>0,c) || throw(ArgumentError("expected n positive integer labels"))
    Int.(collect(c))
end
function frozen_refs(refs,n)
    length(refs)==4 || throw(ArgumentError("exactly four frozen reference partitions required"))
    ntuple(r->FrozenLabels(G.canonical_labels(valid_labels(refs[r],n))),4)
end
labelhash(c)=bytes2hex(sha256(reinterpret(UInt8,Int64.(collect(c)))))
function count_labels(c)
    out=Dict{Int,Int}()
    for h in c;out[h]=get(out,h,0)+1;end
    out
end
function count_cells(c,r)
    out=Dict{Tuple{Int,Int},Int}()
    for i in eachindex(c)
        key=(c[i],r[i]);out[key]=get(out,key,0)+1
    end
    out
end
pairsum(counts)=foldl(checked_add,(paircount(v) for v in values(counts));init=Int128(0))

"""ARI from exact pair aggregates, including its changing row denominator.

All products are checked Int128 arithmetic. n>=2 is required; zero denominator
then denotes the ordinary identical trivial partitions and returns 1, matching
G.ari. A one-observation ARI is undefined in G.ari and is rejected at setup.
"""
function ari_pairs(N::Int128,R::Int128,B::Int128,C::Int128)
    0<=R<=N && 0<=B<=N && 0<=C<=min(R,B) || error("invalid ARI pair aggregates")
    RB=checked_mul(R,B)
    numerator=checked_mul(Int128(2),checked_sub(checked_mul(N,C),RB))
    denominator=checked_sub(checked_mul(N,checked_add(R,B)),checked_mul(Int128(2),RB))
    if denominator==0
        numerator==0 || error("inconsistent degenerate ARI aggregates")
        return 1.
    end
    denominator>0 || error("negative ARI denominator")
    Float64(numerator)/Float64(denominator)
end
function raw_ari(c,r)
    n=length(c);n>=2 || throw(ArgumentError("ARI requires at least two observations"))
    a=valid_labels(c,n);b=valid_labels(r,n);N=paircount(n)
    ari_pairs(N,pairsum(count_labels(a)),pairsum(count_labels(b)),pairsum(count_cells(a,b)))
end
function contrast(N,R,B,C)
    a=ntuple(r->ari_pairs(N,R,B[r],C[r]),4)
    a[4]-(a[1]+a[2]+a[3])/3
end
function raw_score(c,refs)
    length(refs)==4 || throw(ArgumentError("exactly four reference partitions required"))
    a=ntuple(r->raw_ari(c,refs[r]),4)
    a[4]-(a[1]+a[2]+a[3])/3
end

"""Frozen ARI4-minus-mean(ARI1:3) cache, in dimensionless ARI units.

R is the current cluster-pair sum, B_r the fixed reference-pair sum and C_r
the current/reference contingency-pair sum. A single allocation changes R
and every C_r in constant time; both ARI numerator and denominator change.
All pair aggregates remain exact integers, so repeated updates do not
accumulate floating score drift. Score values are recomputed from integers.
The contrast is bounded in [-2,2], making every finite exponential bias proper.
This coordinate does not change or replace any frozen diagnostic classifier.
"""
mutable struct Workspace <: BT.AbstractWorkspace
    n::Int
    total_pairs::Int128
    references::NTuple{4,FrozenLabels}
    reference_hashes::NTuple{4,String}
    reference_pairs::NTuple{4,Int128}
    reference_ari::NTuple{4,NTuple{4,Float64}}
    stateid::UInt
    allocations::Vector{Int}
    counts::Dict{Int,Int}
    cells::NTuple{4,Dict{Tuple{Int,Int},Int}}
    row_pairs::Int128
    cell_pairs::Vector{Int128}
    generation::UInt64
end
function Workspace(s,refs)
    n=length(s.c);n>=2 || throw(ArgumentError("ARI contrast requires at least two observations"))
    N=paircount(n)
    # Conservative setup guard covers all subsequent checked products/sums.
    checked_mul(Int128(4),checked_mul(N,N))
    r=frozen_refs(refs,n)
    B=ntuple(j->pairsum(count_labels(r[j])),4)
    self=ntuple(i->ntuple(j->raw_ari(r[i],r[j]),4),4)
    w=Workspace(n,N,r,ntuple(j->labelhash(r[j]),4),B,self,objectid(s),Int[],Dict{Int,Int}(),
        ntuple(_->Dict{Tuple{Int,Int},Int}(),4),Int128(0),zeros(Int128,4),UInt64(0))
    refresh!(w,s)
end
Workspace(s,a,b,c,d)=Workspace(s,(a,b,c,d))
function BT.refresh!(w::Workspace,s)
    c=valid_labels(s.c,w.n)
    rows=count_labels(c);cells=ntuple(r->count_cells(c,w.references[r]),4)
    R=pairsum(rows);C=Int128[pairsum(a) for a in cells]
    contrast(w.total_pairs,R,w.reference_pairs,C)
    generation=Base.Checked.checked_add(w.generation,UInt64(1))
    w.stateid=objectid(s);w.allocations=c;w.counts=rows;w.cells=cells
    w.row_pairs=R;w.cell_pairs=C;w.generation=generation;w
end
BT.score(w::Workspace)=contrast(w.total_pairs,w.row_pairs,w.reference_pairs,w.cell_pairs)
BT.depends_on(w::Workspace,i)=true
function BT.reference_metadata(w::Workspace)
    (score="ARI(c,reference4)-mean(ARI(c,reference1:3))",units="dimensionless ARI contrast",
     lower_bound=-2.,upper_bound=2.,n=w.n,reference_label_sha256=w.reference_hashes,
     reference_self_ari=w.reference_ari,
     cache="exact checked Int128 row/reference/contingency pair sums; full changing ARI denominators",
     classifier="coordinate only; diagnostic classifier and gray region unchanged")
end
function BT.proposal_change(w::Workspace,i,old,new)
    i isa Integer && 1<=i<=w.n && old isa Integer && new isa Integer && old>0 && new>0 || throw(ArgumentError("invalid ARI allocation change"))
    w.allocations[i]==old || throw(ArgumentError("stale touched ARI allocation"))
    old=Int(old);new=Int(new)
    dr=old==new ? Int128(0) : Int128(get(w.counts,new,0))-Int128(w.counts[old])+1
    dc=ntuple(r->old==new ? Int128(0) :
        Int128(get(w.cells[r],(new,w.references[r][i]),0))-Int128(get(w.cells[r],(old,w.references[r][i]),0))+1,4)
    R=checked_add(w.row_pairs,dr);C=ntuple(r->checked_add(w.cell_pairs[r],dc[r]),4)
    delta=old==new ? 0. : contrast(w.total_pairs,R,w.reference_pairs,C)-score(w)
    (delta=delta,data=(index=Int(i),old=old,new=new,generation=w.generation,row_delta=dr,cell_deltas=dc))
end
function decrement!(counts,key)
    left=counts[key]-1
    left==0 ? delete!(counts,key) : (counts[key]=left)
end
function BT.accept_change!(w::Workspace,i,old,new,change)
    d=change.data
    (i,old,new,w.generation)==(d.index,d.old,d.new,d.generation) && w.allocations[i]==old || throw(ArgumentError("stale or mismatched ARI proposal"))
    R=checked_add(w.row_pairs,d.row_delta);C=Int128[checked_add(w.cell_pairs[r],d.cell_deltas[r]) for r in 1:4]
    generation=Base.Checked.checked_add(w.generation,UInt64(1))
    # All fallible structural/overflow checks precede mutable cache changes.
    if old!=new
        get(w.counts,old,0)>=1 && all(r->get(w.cells[r],(old,w.references[r][i]),0)>=1,1:4) || error("invalid ARI count cache")
        decrement!(w.counts,old);w.counts[new]=get(w.counts,new,0)+1
        for r in 1:4
            a=(old,w.references[r][i]);b=(new,w.references[r][i])
            decrement!(w.cells[r],a);w.cells[r][b]=get(w.cells[r],b,0)+1
        end
    end
    w.row_pairs=R;w.cell_pairs=C;w.allocations[i]=new;w.generation=generation;w
end

end
