mutable struct MarginalState{D,L}
    model::Model{D,L}
    S::Vector{Int}
    c::Vector{Int}
    clusters::Vector{Cluster{D,L}}
    active::Vector{Int}
    F::Matrix{Float64}
    Q::Matrix{Float64}
    ld::Float64                    # dimensionless determinant
    bound::Float64
    streams::Streams
    counters::Counters
    workf::Vector{Float64}
    workv::Vector{Float64}
    workw::Vector{Float64}
    predictive_norms::Vector{Float64}
end
mutable struct ConditionalState{D,L}
    base::MarginalState{D,L}
    unused::Vector{SVector{D,Float64}}
    covarianceL::Vector{SMatrix{D,D,Float64,L}} # active order then unused
    weights::Vector{Float64}
end
core(s::MarginalState)=s
core(s::ConditionalState)=s.base
function rebuild!(s::MarginalState)
    s.F=feature_matrix(s.model,s.S,[s.clusters[h].x for h in s.active])
    s.Q,s.ld=orthobasis(s.F,s.counters)
    isfinite(s.ld) || error("zero-target occupied configuration")
    s.bound=envelope(s.model.spectrum,s.S)
    resize!(s.workf,length(s.S)); resize!(s.workv,length(s.S))
    s.counters.projector_rebuilds+=1
    s
end
function newslot!(s::MarginalState,x;excluded=Int[])
    h=findfirst(i->s.clusters[i].n==0 && !(i in excluded),eachindex(s.clusters))
    if isnothing(h)
        push!(s.clusters,Cluster(s.model,x)); h=length(s.clusters)
        grow_streams!(s.streams,h)
    else
        s.clusters[h].x=x
    end
    h
end
function spectral_update!(s::MarginalState)
    m=s.model; old=s.S
    logz=logbeta_draw(s.streams.spectral,m.a*length(old),length(m.y))
    J=old[sample_J(s.streams.spectral,s.F,s.counters;Qcache=s.Q)]
    s.S=shell_draw(s.streams.spectral,m.spectrum,J,m.a*logz)
    s.counters.spectral_added+=length(setdiff(s.S,old))
    s.counters.spectral_removed+=length(setdiff(old,s.S))
    rebuild!(s)
end
function allocation!(s::MarginalState,i;envelope_multiplier=1.0,maxtries=10^7)
    envelope_multiplier>=1 || throw(ArgumentError("envelope multiplier must be >=1"))
    m=s.model; rng=s.streams.obs[i]; y=m.y[i]; old=s.c[i]
    remove!(s.clusters[old],y)
    if s.clusters[old].n==0
        s.clusters[old].x=zero(s.clusters[old].x)
        s.clusters[old].scaleL=m.omegaL
        s.clusters[old].logdet=m.logdetomega
        s.clusters[old].logpredconst=0.0
        deleteat!(s.active,findfirst(==(old),s.active))
        s.counters.deletions+=1
        rebuild!(s)
    else
        refresh!(s.clusters[old],m,s.predictive_norms)
    end
    q=length(s.active); newpossible=q<length(s.S)
    resize!(s.workw,q+Int(newpossible))
    for (j,h) in enumerate(s.active); s.workw[j]=predictive_mass(s.clusters[h],m,y); end
    b=s.bound*envelope_multiplier
    if newpossible; s.workw[end]=log(m.a)+log(b)-m.logvolume; end
    mx=maximum(s.workw)
    isfinite(mx) || error("all allocation masses nonfinite")
    for j in eachindex(s.workw); s.workw[j]=exp(s.workw[j]-mx); end
    for _ in 1:maxtries
        # Restart this categorical draw after every rejected new-location candidate.
        branch=draw_weights(rng,s.workw)
        if branch<=q
            h=s.active[branch]; add!(s.clusters[h],y); refresh!(s.clusters[h],m,s.predictive_norms); s.c[i]=h
            return h
        end
        s.counters.new_attempts+=1
        x=student_draw(rng,y,m.omegaL,m.tau-length(y)+1)
        if !inside(m,x)
            s.counters.new_outside+=1; continue
        end
        features!(s.workf,m,s.S,x)
        r=residual!(s.workv,s.Q,s.workf)
        if rand(rng)<bounded_ratio(r,b,s.counters)
            h=newslot!(s,x)
            add!(s.clusters[h],y); refresh!(s.clusters[h],m,s.predictive_norms)
            push!(s.active,h); s.c[i]=h
            s.counters.births+=1; s.counters.new_accepts+=1
            rebuild!(s)
            return h
        end
        s.counters.new_rejects+=1
    end
    error("allocation rejection retry limit exceeded for observation $i; run aborted")
end
function logdet_features(F,ct)
    isempty(F) && return 0.0
    size(F,2)>size(F,1) && return -Inf
    f=qr(F); ds=abs.(diag(f.R))
    if minimum(ds)<=1e-10*max(1.0,maximum(ds))
        ct.near_rank+=1
        ds=svdvals(F)
    end
    any(iszero,ds) ? -Inf : 2sum(log,ds)
end
function location_accepts!(s,proposals,proposal_features,F;full=false)
    ld=logdet_features(F,s.counters); isfinite(ld) || error("invalid current location determinant")
    for (j,h) in enumerate(s.active)
        s.counters.loc_attempts+=1
        x=proposals[j]
        if !inside(s.model,x)
            s.counters.loc_outside+=1; continue
        end
        old=copy(view(F,:,j)); F[:,j].=view(proposal_features,:,j)
        ldnew=logdet_features(F,s.counters)
        if log(rand(s.streams.accept))<ldnew-ld
            s.clusters[h].x=x; refresh!(s.clusters[h],s.model,s.predictive_norms)
            s.counters.loc_accepts+=1; ld=ldnew
        else
            F[:,j].=old
        end
    end
    rebuild!(s)
end
function marginal_locations!(s::MarginalState{D,L},ex) where {D,L}
    k=length(s.active); m=s.model
    proposals=Vector{SVector{D,Float64}}(undef,k)
    PF=Matrix{Float64}(undef,length(s.S),k)
    foreach_chunk(k,ex) do range
        for j in range
            h=s.active[j]; c=s.clusters[h]
            R=m.omega+c.scatter
            factor=SMatrix{D,D,Float64,L}(cholesky(Symmetric(R/c.n)).L)
            proposals[j]=student_draw(s.streams.loc[h],c.mean,factor,m.tau+c.n-D)
            features!(view(PF,:,j),m,s.S,proposals[j])
        end
    end
    location_accepts!(s,proposals,PF,s.F)
end
function marginal_collapsed!(s::MarginalState,ex=Execution())
    for i in eachindex(s.c); allocation!(s,i); end
    marginal_locations!(s,ex)
    spectral_update!(s)
    s
end
function reconstruct!(s::ConditionalState{D,L},activecov=nothing) where {D,L}
    b=s.base; m=b.model; k=length(b.active)
    pts=palm_sample(b.streams.palm,m,b.S,[b.clusters[h].x for h in b.active],b.counters)
    s.unused=pts[k+1:end]
    if isnothing(activecov)
        activecov=[iw_factor(b.streams.cov[h],m.tau+b.clusters[h].n,b.clusters[h].scaleL) for h in b.active]
    end
    s.covarianceL=copy(activecov)
    for _ in s.unused; push!(s.covarianceL,iw_factor(b.streams.palm,m.tau,m.omegaL)); end
    resize!(s.weights,length(b.S))
    for j in eachindex(s.weights)
        count=j<=k ? b.clusters[b.active[j]].n : 0
        s.weights[j]=loggamma_draw(b.streams.weights,count+m.a)
    end
    mx=maximum(s.weights)
    for j in eachindex(s.weights); s.weights[j]=exp(s.weights[j]-mx); end
    s.weights./=sum(s.weights)
    s
end
function conditional_allocations!(s::ConditionalState{D,L},ex) where {D,L}
    b=s.base; m=b.model; oldactive=copy(b.active)
    xs=vcat([b.clusters[h].x for h in oldactive],s.unused)
    ids=copy(oldactive)
    for x in s.unused; push!(ids,newslot!(b,x;excluded=ids)); end
    # Every old full-configuration component has an ID during this allocation block.
    M=length(xs); newc=similar(b.c)
    norms=[log(s.weights[j])-D/2*log(2pi)-sum(log,diag(s.covarianceL[j])) for j in 1:M]
    foreach_chunk(length(b.c),ex) do range
        w=Vector{Float64}(undef,M) # task-owned scratch, never indexed by thread id
        for i in range
            for j in 1:M
                z=LowerTriangular(s.covarianceL[j])\(m.y[i]-xs[j])
                w[j]=norms[j]-sum(abs2,z)/2
            end
            newc[i]=ids[draw_logweights!(b.streams.obs[i],w)]
        end
    end
    for h in ids
        c=b.clusters[h]; c.n=0; c.mean=zero(c.mean); c.scatter=zero(c.scatter)
    end
    b.c=newc
    for i in eachindex(newc); add!(b.clusters[newc[i]],m.y[i]); end
    b.active=Int[h for h in ids if b.clusters[h].n>0]
    b.counters.births+=count(h->!(h in oldactive),b.active)
    b.counters.unused_occupied+=count(h->!(h in oldactive),b.active)
    b.counters.deletions+=count(h->!(h in b.active),oldactive)
    for h in b.active; refresh!(b.clusters[h],m,b.predictive_norms); end
    # Reorder the full configuration so active proposals occupy its first columns.
    inactive=Int[h for h in ids if b.clusters[h].n==0]
    fullxs=vcat([b.clusters[h].x for h in b.active],[b.clusters[h].x for h in inactive])
    for h in inactive
        b.clusters[h].x=zero(b.clusters[h].x)
        b.clusters[h].scaleL=m.omegaL
        b.clusters[h].logdet=m.logdetomega
        b.clusters[h].logpredconst=0.0
    end
    fullxs
end
function conditional_locations!(s::ConditionalState{D,L},fullxs,ex) where {D,L}
    b=s.base; m=b.model; k=length(b.active)
    proposals=Vector{SVector{D,Float64}}(undef,k)
    covs=Vector{SMatrix{D,D,Float64,L}}(undef,k)
    PF=Matrix{Float64}(undef,length(b.S),k)
    foreach_chunk(k,ex) do range
        for j in range
            h=b.active[j]; c=b.clusters[h]
            covs[j]=iw_factor(b.streams.cov[h],m.tau+c.n,c.scaleL)
            found=false
            for _ in 1:10^7
                x=mvnormal_draw(b.streams.loc[h],c.mean,covs[j]/sqrt(c.n))
                if inside(m,x); proposals[j]=x; found=true; break; end
            end
            found || error("truncated Gaussian proposal retry limit exceeded")
            features!(view(PF,:,j),m,b.S,proposals[j])
        end
    end
    F=feature_matrix(m,b.S,fullxs)
    location_accepts!(b,proposals,PF,F;full=true)
    covs
end
function conditional_notes!(s::ConditionalState,ex=Execution())
    fullxs=conditional_allocations!(s,ex)
    covs=conditional_locations!(s,fullxs,ex)
    spectral_update!(s.base)
    reconstruct!(s,covs)
    s
end
sweep!(s::MarginalState,ex=Execution())=marginal_collapsed!(s,ex)
sweep!(s::ConditionalState,ex=Execution())=conditional_notes!(s,ex)
