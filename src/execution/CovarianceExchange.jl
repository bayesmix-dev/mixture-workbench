# Included inside Benchmark. Only the IW scale varies across replicas.
# All expressions below omit terms constant in that scale, not in the state.
function covariance_logpart(s, m, method)
    total=0.0
    for c in s.clusters
        c.n==0 && continue
        if method==:mfm
            delta=c.mean-s.mu0
            scatter=c.scatter+(s.kappa*c.n/(s.kappa+c.n))*(delta*delta')
        else
            delta=c.mean-c.x
            scatter=c.scatter+c.n*(delta*delta')
        end
        ld=2sum(log,diag(cholesky(Symmetric(m.omega+scatter)).L))
        total+=m.tau/2*m.logdetomega-(m.tau+c.n)/2*ld
    end
    total
end
function exchange_logratio(a,b,ma,mb,method)
    covariance_logpart(b,ma,method)+covariance_logpart(a,mb,method)-
        covariance_logpart(a,ma,method)-covariance_logpart(b,mb,method)
end
function rebind_covariance!(s,m,method)
    s.model=m
    for h in eachindex(s.clusters)
        if method==:mfm
            MF.refresh!(s,h)
        else
            G.refresh!(s.clusters[h],m)
        end
    end
    s
end
mutable struct CovarianceExchange
    states::Vector{Any}
    models::Vector{Any}
    works::Vector{Any}
    rngs::Vector{Xoshiro}
    swaprng::Xoshiro
    target::Int
    attempts::Vector{Int}
    accepts::Vector{Int}
    tags::Vector{Int}
    target_visits::Vector{Int}
    smstats::Vector{Int}
end
function initialize_exchange(m,init,elic,method,seed,ladder,targetscale)
    target=findfirst(==(targetscale),ladder)
    isnothing(target) && error("target covariance scale must occur in ladder")
    models=Any[G.Model(m.y,m.truth,m.lower,m.upper,m.spectrum;
        a=m.a,tau=m.tau,omega=m.omega*(x/targetscale),seed=m.seed) for x in ladder]
    starts=[j==target ? init : G.starting_configuration(mm,
        min(length(m.y),length(m.spectrum.kind),round(Int,15-13*(j-1)/(length(ladder)-1))),G.seedof(seed,999,j))
        for (j,mm) in enumerate(models)]
    states=Any[setup(mm,starts[j],method,G.seedof(seed,j),elic.radius;
        strauss_beta=get(elic,:strauss_beta,10.),strauss_logalpha=get(elic,:strauss_logalpha,-length(m.y)/20))
        for (j,mm) in enumerate(models)]
    works=Any[method==:dpp_marginal_sm ? SM.Workspace(mm) : nothing for mm in models]
    CovarianceExchange(states,models,works,[Xoshiro(G.seedof(seed,77,j)) for j in eachindex(models)],
        Xoshiro(G.seedof(seed,889)),target,zeros(Int,length(ladder)-1),zeros(Int,length(ladder)-1),
        collect(eachindex(ladder)),zeros(Int,length(ladder)),zeros(Int,4))
end
function exchange_step!(e,method,it;every=5)
    for j in eachindex(e.states)
        # Auxiliary chains need only update when an exchange is scheduled.
        # Different update frequencies still preserve the product target.
        j==e.target || it%every==0 || continue
        s=e.states[j]
        if method==:dpp_marginal_sm
            G.marginal_collapsed!(s)
            p=lock(SM_LOCK) do; SM.attempt!(e.works[j],s,e.rngs[j]); end
            if p!==nothing
                h=p.kind==:split ? 1 : 2
                e.smstats[h]+=1
                p.outcome==:accepted && (e.smstats[h+2]+=1)
            end
        else
            step!(s,method,e.works[j],e.rngs[j])
        end
    end
    if it%every==0
        # Randomize scan direction; each adjacent MH kernel preserves the product target.
        pairs=collect(eachindex(e.attempts)); rand(e.swaprng,Bool) && reverse!(pairs)
        for j in pairs
            e.attempts[j]+=1
            lr=exchange_logratio(e.states[j],e.states[j+1],e.models[j],e.models[j+1],method)
            if log(rand(e.swaprng))<lr
                e.states[j],e.states[j+1]=e.states[j+1],e.states[j]
                e.tags[j],e.tags[j+1]=e.tags[j+1],e.tags[j]
                rebind_covariance!(e.states[j],e.models[j],method)
                rebind_covariance!(e.states[j+1],e.models[j+1],method)
                e.accepts[j]+=1
            end
        end
        if method in (:strauss_general,:dpp_general)
            # The exchange integrates out covariances. Restore them even on rejection:
            # retaining old marks after a collapsed rejection is not a valid kernel.
            for s in e.states
                for (j,c) in enumerate(s.clusters)
                    G.refresh!(c,s.model)
                    s.covs[j]=G.iw_factor(s.rng,s.model.tau+c.n,c.scaleL)
                end
            end
        end
    end
    e.target_visits[e.tags[e.target]]+=1
    e.states[e.target]
end
