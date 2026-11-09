mutable struct Cluster{D,L}
    n::Int
    mean::SVector{D,Float64}
    scatter::SMatrix{D,D,Float64,L}
    x::SVector{D,Float64}
    scaleL::SMatrix{D,D,Float64,L}
    logdet::Float64
    logpredconst::Float64
end
function Cluster(m::Model{D,L},x) where {D,L}
    Cluster(0,zero(x),zero(m.omega),x,m.omegaL,m.logdetomega,0.0)
end
function add!(c::Cluster,y)
    n=c.n; delta=y-c.mean
    c.mean+=delta/(n+1)
    c.scatter+=(n/(n+1))*(delta*delta')
    c.n+=1; c
end
function remove!(c::Cluster,y)
    c.n>0 || error("removing from empty cluster")
    if c.n==1
        c.n=0; c.mean=zero(c.mean); c.scatter=zero(c.scatter)
    else
        n=c.n; delta=y-c.mean
        c.mean-=delta/(n-1)
        c.scatter-=(n/(n-1))*(delta*delta'); c.n-=1
    end
    c
end
@inline function predictive_normalizer(m::Model{D},n) where D
    df=m.tau+n-D+1
    log(n+m.a)+loggamma((df+D)/2)-loggamma(df/2)-D/2*log(pi)
end
function refresh!(c::Cluster,m::Model{D},norms=nothing) where D
    delta=c.mean-c.x
    mat=m.omega+c.scatter+c.n*(delta*delta')
    c.scaleL=cholesky(Symmetric(mat)).L
    c.logdet=2sum(log,diag(c.scaleL))
    norm=isnothing(norms) ? predictive_normalizer(m,c.n) : norms[c.n+1]
    c.logpredconst=norm-c.logdet/2
    c
end
@inline function predictive_mass(c::Cluster,m::Model{D},y) where D
    z=LowerTriangular(c.scaleL)\(y-c.x)
    c.logpredconst-(m.tau+c.n+1)/2*log1p(sum(abs2,z))
end
function log_marginal(c::Cluster,m::Model{D}) where D
    n=c.n
    n==0 && return 0.0
    -n*D/2*log(pi)+m.tau/2*m.logdetomega-(m.tau+n)/2*c.logdet+
        logmultigamma((m.tau+n)/2,D)-logmultigamma(m.tau/2,D)
end
