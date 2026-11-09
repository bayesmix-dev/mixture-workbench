module DPPLikelihoodTempering

using GeneralDPPMixtures, LinearAlgebra, Random, StaticArrays
const G = GeneralDPPMixtures

export sweep!, allocation!, locations!, loglikelihood, swap_logratio,
       logtarget, new_location_envelope

"""Sum of covariance-collapsed cluster log likelihoods, without prior terms."""
loglikelihood(s::G.MarginalState)=sum(G.log_marginal(s.clusters[h],s.model) for h in s.active)
function checkbeta(beta)
    isfinite(beta) && 0<=beta<=1 || throw(ArgumentError("inverse temperature must lie in [0,1]"))
    Float64(beta)
end
"""The cold prior is fixed; only the collapsed likelihood is powered."""
function logtarget(s::G.MarginalState,beta)
    b=checkbeta(beta)
    G.collapsed_logtarget(s)+(b-1)*loglikelihood(s)
end
function swap_logratio(a::G.MarginalState,b::G.MarginalState,beta_a,beta_b)
    ba=checkbeta(beta_a);bb=checkbeta(beta_b)
    (ba-bb)*(loglikelihood(b)-loglikelihood(a))
end

"""
A rejection envelope for the singleton-location likelihood power. `offset` is
its log branch mass excluding log(a)+log(DPP envelope). A proper powered Student
law uses offset=log(Z)-log(volume); uniform-box rejection uses offset=beta*Lmax.
The smaller valid bound is selected deterministically from the fixed model.
"""
function new_location_envelope(m::G.Model{D},beta) where D
    b=checkbeta(beta)
    logmax=-D/2*log(pi)-m.logdetomega/2+
        G.logmultigamma((m.tau+1)/2,D)-G.logmultigamma(m.tau/2,D)
    df=b*(m.tau+1)-D
    uniform_offset=b*logmax
    if df>0
        origin=zero(first(m.y))
        logqmax=G.student_logdensity(origin,origin,m.omegaL,df)
        logZ=b*logmax-logqmax
        student_offset=logZ-m.logvolume
        if student_offset<uniform_offset
            return (mode=:student,df=df,logmax=logmax,logZ=logZ,offset=student_offset,beta=b)
        end
    end
    (mode=:uniform,df=df,logmax=logmax,logZ=NaN,offset=uniform_offset,beta=b)
end

function allocation!(s::G.MarginalState,i,beta;maxtries=10^7)
    b=checkbeta(beta)
    b==1 && return G.allocation!(s,i;maxtries)
    _allocation!(s,i,b,new_location_envelope(s.model,b);maxtries)
end
function _allocation!(s::G.MarginalState,i,beta,law;maxtries=10^7)
    m=s.model;rng=s.streams.obs[i];y=m.y[i];old=s.c[i]
    G.remove!(s.clusters[old],y)
    if s.clusters[old].n==0
        s.clusters[old].x=zero(s.clusters[old].x)
        s.clusters[old].scaleL=m.omegaL
        s.clusters[old].logdet=m.logdetomega
        s.clusters[old].logpredconst=0.
        deleteat!(s.active,findfirst(==(old),s.active));s.counters.deletions+=1
        G.rebuild!(s)
    else
        G.refresh!(s.clusters[old],m,s.predictive_norms)
    end
    q=length(s.active);newpossible=q<length(s.S)
    resize!(s.workw,q+Int(newpossible))
    for (j,h) in enumerate(s.active)
        c=s.clusters[h];weight=log(c.n+m.a)
        s.workw[j]=weight+beta*(G.predictive_mass(c,m,y)-weight)
    end
    if newpossible;s.workw[end]=log(m.a)+log(s.bound)+law.offset;end
    mx=maximum(s.workw);isfinite(mx) || error("all tempered allocation masses nonfinite")
    for j in eachindex(s.workw);s.workw[j]=exp(s.workw[j]-mx);end
    for _ in 1:maxtries
        # Every rejected location restarts the full categorical draw. Holding
        # the new-component branch fixed would change the allocation target.
        branch=G.draw_weights(rng,s.workw)
        if branch<=q
            h=s.active[branch];G.add!(s.clusters[h],y)
            G.refresh!(s.clusters[h],m,s.predictive_norms);s.c[i]=h;return h
        end
        s.counters.new_attempts+=1
        x=law.mode==:student ? G.student_draw(rng,y,m.omegaL,law.df) : G.uniform_point(rng,m)
        if !G.inside(m,x);s.counters.new_outside+=1;continue;end
        G.features!(s.workf,m,s.S,x)
        residual=G.residual!(s.workv,s.Q,s.workf)
        probability=G.bounded_ratio(residual,s.bound,s.counters)
        if law.mode==:uniform && beta>0
            # The singleton marginal density is maximized at x=y.
            z=LowerTriangular(m.omegaL)\(y-x)
            logrelative=-(m.tau+1)/2*log1p(sum(abs2,z))
            probability*=exp(beta*logrelative)
        end
        if rand(rng)<probability
            h=G.newslot!(s,x);G.add!(s.clusters[h],y)
            G.refresh!(s.clusters[h],m,s.predictive_norms)
            push!(s.active,h);s.c[i]=h
            s.counters.births+=1;s.counters.new_accepts+=1
            G.rebuild!(s);return h
        end
        s.counters.new_rejects+=1
    end
    error("tempered allocation rejection retry limit exceeded for observation $i; run aborted")
end

"""
Location MH updates conditional on the partition and spectrum. Where the powered
likelihood is integrable over Euclidean space, its exact Student law proposes a
location and the acceptance ratio contains only DPP geometry. Otherwise use a
uniform-box proposal and include the powered likelihood difference explicitly.
"""
function locations!(s::G.MarginalState{D,L},beta,ex=G.Execution()) where {D,L}
    b=checkbeta(beta)
    b==1 && return G.marginal_locations!(s,ex)
    m=s.model;k=length(s.active)
    proposals=Vector{SVector{D,Float64}}(undef,k)
    student=fill(false,k) # ordinary Bool storage: disjoint thread writes are safe
    G.foreach_chunk(k,ex) do range
        for j in range
            # Explicit closure-local bindings prevent later acceptance-loop
            # names from becoming shared boxed variables across worker tasks.
            local h=s.active[j]
            local c=s.clusters[h]
            local df=b*(m.tau+c.n)-D
            if df>0
                local factor=SMatrix{D,D,Float64,L}(cholesky(Symmetric((m.omega+c.scatter)/c.n)).L)
                proposals[j]=G.student_draw(s.streams.loc[h],c.mean,factor,df)
                student[j]=true
            else
                proposals[j]=G.uniform_point(s.streams.loc[h],m)
            end
        end
    end
    F=copy(s.F);ld=G.logdet_features(F,s.counters)
    isfinite(ld) || error("invalid current tempered location determinant")
    f=zeros(length(s.S))
    for (j,h) in enumerate(s.active)
        s.counters.loc_attempts+=1;x=proposals[j]
        if !G.inside(m,x);s.counters.loc_outside+=1;continue;end
        old=copy(view(F,:,j));G.features!(f,m,s.S,x);F[:,j].=f
        ldnew=G.logdet_features(F,s.counters);ratio=ldnew-ld
        if !student[j] && b>0
            c=s.clusters[h];delta=c.mean-x
            R=m.omega+c.scatter+c.n*(delta*delta')
            newlogdet=2sum(log,diag(cholesky(Symmetric(R)).L))
            ratio-=b*(m.tau+c.n)/2*(newlogdet-c.logdet)
        end
        if log(rand(s.streams.accept))<ratio
            s.clusters[h].x=x;G.refresh!(s.clusters[h],m,s.predictive_norms)
            s.counters.loc_accepts+=1;ld=ldnew
        else
            F[:,j].=old
        end
    end
    G.rebuild!(s)
end

"""
One exact sweep for prior(S,c,x) * product_h marginal_likelihood_h^beta.
At beta=1 this calls the original kernel directly, preserving its random draws.
The spectral conditional is unchanged because likelihood is independent of S.
"""
function sweep!(s::G.MarginalState,beta,ex=G.Execution();maxtries=10^7)
    b=checkbeta(beta)
    b==1 && return G.marginal_collapsed!(s,ex)
    law=new_location_envelope(s.model,b)
    for i in eachindex(s.c);_allocation!(s,i,b,law;maxtries);end
    locations!(s,b,ex)
    G.spectral_update!(s)
    # The spectral rebuild can grow Float64 scratch with uninitialized tails.
    # These buffers carry no Markov state and are overwritten before use, but
    # leaving heap contents here makes checkpoints/thread-equivalence depend on
    # allocation history. Reset scratch without changing any random draws.
    fill!(s.workf,0.)
    fill!(s.workv,0.)
    fill!(s.workw,0.)
    s
end

end
