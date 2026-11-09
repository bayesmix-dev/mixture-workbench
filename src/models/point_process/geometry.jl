function features!(out,m::Model{D},S,x) where D
    u=(x-m.lower).*m.invwidth
    @inbounds for (j,mode) in enumerate(S)
        kind=m.spectrum.kind[mode]
        if kind==0
            out[j]=1.0
        else
            z=0.0
            for e in 1:D; z+=m.spectrum.frequencies[e,mode]*u[e]; end
            out[j]=sqrt(2.0)*(kind==1 ? cospi(2z) : sinpi(2z))
        end
    end
    out
end
function feature_matrix(m,S,xs)
    F=Matrix{Float64}(undef,length(S),length(xs))
    for h in eachindex(xs); features!(view(F,:,h),m,S,xs[h]); end
    F
end
function envelope(spec,S)
    n0=0; pairs=Set{Int}()
    for j in S
        spec.kind[j]==0 ? (n0+=1) : push!(pairs,spec.kind[j]==1 ? j : j-1)
    end
    Float64(n0+2length(pairs))  # dimensionless bound; divide by volume for intensity
end
function orthobasis(F::Matrix{Float64},ct::Counters)
    M,k=size(F)
    k==0 && return zeros(M,0),0.0
    k<=M || return zeros(M,0),-Inf
    fac=qr(F)
    diagonal=abs.(diag(fac.R))
    if minimum(diagonal)<=1e-10*max(1.0,maximum(diagonal))
        ct.near_rank+=1
        sv=svd(F)
        minimum(sv.S)==0 && return sv.U,-Inf
        return sv.U,2sum(log,sv.S)
    end
    Q=fac.Q*Matrix{Float64}(I,M,k)
    Q,2sum(log,diagonal)
end
# Explicit residual norm instead of subtracting squared lengths; two passes.
function residual!(v,Q,f)
    copyto!(v,f)
    @inbounds for _ in 1:2, h in axes(Q,2)
        coeff=0.0
        for j in eachindex(v); coeff+=Q[j,h]*v[j]; end
        for j in eachindex(v); v[j]-=Q[j,h]*coeff; end
    end
    sum(abs2,v)
end
function bounded_ratio(r,b,ct::Counters)
    if r>b
        r-b <= 256eps(Float64)*max(1.0,b) || error("Residual exceeds envelope: $r > $b")
        ct.clamps+=1; return 1.0
    end
    r>=0 || error("negative squared norm")
    r/b
end
function sample_J(rng,F,ct= Counters(); Qcache=nothing)
    M,k=size(F)
    M==k && return collect(1:k)
    Q,ld=isnothing(Qcache) ? orthobasis(F,ct) : (copy(Qcache),0.0)
    isfinite(ld) || error("rank-deficient current occupied features")
    chosen=Int[]; weights=zeros(M); v=zeros(k)
    for s in k:-1:1
        for j in 1:M; weights[j]=sum(abs2,view(Q,j,1:s)); end
        for j in chosen; weights[j]=0.0; end
        row=draw_weights(rng,weights); push!(chosen,row)
        s==1 && break
        for h in 1:s; v[h]=Q[row,h]; end
        normv=norm(view(v,1:s))
        v[s]+=copysign(normv,v[s])
        scale=2/sum(abs2,view(v,1:s))
        # Right Householder application; last column discarded. No dense reflector.
        for j in 1:M
            coeff=0.0
            for h in 1:s; coeff+=Q[j,h]*v[h]; end
            coeff*=scale
            for h in 1:s-1; Q[j,h]-=coeff*v[h]; end
        end
        Q[row,1:s-1].=0.0
    end
    sort!(chosen)
end
function palm_sample(rng,m,S,occupied,ct=Counters();maxtries=10^7)
    F=feature_matrix(m,S,occupied); Q,ld=orthobasis(F,ct)
    isfinite(ld) || error("Palm conditioning has zero target")
    b=envelope(m.spectrum,S); M=length(S)
    out=copy(occupied); f=zeros(M); v=zeros(M)
    while length(out)<M
        found=false
        for _ in 1:maxtries
            ct.palm_attempts+=1
            x=uniform_point(rng,m); features!(f,m,S,x)
            r=residual!(v,Q,f)
            if rand(rng)<bounded_ratio(r,b,ct)
                r>0 || error("zero residual selected")
                push!(out,x); Q=hcat(Q,v/sqrt(r)); found=true; break
            end
        end
        found || error("Palm rejection retry limit exceeded; run aborted")
    end
    out
end
