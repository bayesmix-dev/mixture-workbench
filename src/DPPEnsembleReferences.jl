module DPPEnsembleReferences

using GeneralDPPMixtures
import ..DPPReallocation as RA
import ..DPPMultiReallocation as MR
import ..DPPGlobalSplitMerge as GS
const G=GeneralDPPMixtures

export conditional_workspaces

function copied_bank(w::MR.Workspace,live,n)
    size(w.whitened,2)==n || throw(ArgumentError("workspace observations differ from ensemble"))
    all(ref->length(ref)==n && all(>(0),ref),w.references) || throw(ArgumentError("invalid frozen reference bank"))
    # Only the fixed observation whitening is shared. Every reference vector,
    # including pre-MCMC training references, belongs to this conditional call.
    refs=vcat([copy(ref) for ref in w.references],[copy(ref) for ref in live])
    MR.Workspace(refs,w.whitened,w.blocks,w.conflict_policy)
end
conditional_workspace(w::MR.Workspace,live,n)=copied_bank(w,live,n)
conditional_workspace(w::GS.Workspace,live,n)=GS.Workspace(copied_bank(w.bank,live,n))
# The original two-cluster kernel has no reference-bank dependence. Its mutable
# scratch must already be private to the replica, as in the ordinary driver.
conditional_workspace(w::RA.Workspace,live,n)=w
conditional_workspace(::Nothing,live,n)=nothing
owns_live_reference(w::MR.Workspace,labels)=any(ref->ref===labels,w.references)
owns_live_reference(w::GS.Workspace,labels)=owns_live_reference(w.bank,labels)
owns_live_reference(::RA.Workspace,labels)=false
owns_live_reference(::Nothing,labels)=false

"""
    conditional_workspaces(pairw,multiw,globalw,states,j;partners=nothing)

Return `(pair,multi,global)` workspaces for ONE sequential update of replica j.
Each reference-bank workspace contains copies of its original frozen references
and canonical partition copies from the specified OTHER replicas. The default
partners are all indices except j. Partner selection must be fixed or independent
of replica j; it must not be selected using the active replica's current state.

Hold the returned banks fixed for the complete forward/reverse probability
calculation. Never refresh them inside an MH proposal. Other replicas must remain
unchanged until the update of j has finished. These conditional kernels preserve
the product posterior; independently updating several replicas against banks
that are refreshed concurrently is not covered by this construction.

Own live state references and aliased state/label storage are rejected. Original
training references may coincide with any current partition if frozen before
sampling. The fixed data whitening is reused without mutation; reference vectors
are copied to avoid subsequent partner updates changing a proposal's law.
"""
function conditional_workspaces(pairw,multiw,globalw,states,j;partners=nothing)
    j isa Integer && 1<=j<=length(states) || throw(ArgumentError("invalid active replica index"))
    selected=partners===nothing ? [i for i in eachindex(states) if i!=j] : collect(partners)
    all(i->i isa Integer && 1<=i<=length(states),selected) || throw(ArgumentError("invalid partner index"))
    length(unique(selected))==length(selected) || throw(ArgumentError("duplicate partner indices"))
    j in selected && throw(ArgumentError("own live partition cannot be a reference"))
    n=length(states[j].c)
    any(w->owns_live_reference(w,states[j].c),(pairw,multiw,globalw)) &&
        throw(ArgumentError("a base reference bank aliases the active live partition"))
    # Distinct coordinates of a product chain require distinct mutable storage.
    for a in eachindex(states),b in eachindex(states)
        a<b || continue
        (states[a]===states[b] || states[a].c===states[b].c) &&
            throw(ArgumentError("ensemble replicas alias mutable state or label storage"))
    end
    all(i->length(states[i].c)==n,selected) || throw(ArgumentError("partner observations differ"))
    live=[G.canonical_labels(states[i].c) for i in selected]
    (conditional_workspace(pairw,live,n),conditional_workspace(multiw,live,n),
        conditional_workspace(globalw,live,n))
end

end
