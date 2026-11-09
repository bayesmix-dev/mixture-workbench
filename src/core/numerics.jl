function draw_weights(rng,w)
    total=sum(w); total>0 && isfinite(total) || error("invalid categorical weights")
    u=rand(rng)*total; accum=0.0
    for h in eachindex(w)
        accum+=w[h]
        u<accum && return h
    end
    findlast(>(0),w)::Int
end
function draw_logweights!(rng,w)
    mx=maximum(w); isfinite(mx) || error("all categorical log weights nonfinite")
    for j in eachindex(w); w[j]=exp(w[j]-mx); end
    draw_weights(rng,w)
end
@inline function mvnormal_draw(rng,mu::SVector{D},L::SMatrix{D,D}) where D
    mu+L*SVector{D,Float64}(ntuple(_->randn(rng),D))
end
@inline function student_draw(rng,mu::SVector{D},L,df) where D
    z=SVector{D,Float64}(ntuple(_->randn(rng),D))
    mu+(L*z)/sqrt(rand(rng,Chisq(df))) # L L' = df * scale
end
@inline logmultigamma(a,d)=d*(d-1)/4*log(pi)+sum(loggamma(a+(1-j)/2) for j in 1:d)
@inline function student_logdensity(y,mu,L,df)  # L L' = df * scale
    d=length(y); z=LowerTriangular(L)\(y-mu)
    loggamma((df+d)/2)-loggamma(df/2)-d/2*log(pi)-sum(log,diag(L))-(df+d)/2*log1p(sum(abs2,z))
end
function iw_factor(rng,df,L::SMatrix{D,D,T,N}) where {D,T,N}
    # Bartlett A A' ~ Wishart(df,I), Sigma = L (A A')^-1 L'.
    A=zeros(MMatrix{D,D,Float64,N})
    for j in 1:D
        A[j,j]=sqrt(rand(rng,Chisq(df-j+1)))
        for i in j+1:D; A[i,j]=randn(rng); end
    end
    C=L/UpperTriangular(SMatrix(A)')
    SMatrix{D,D,Float64,N}(cholesky(Symmetric(C*C')).L)
end
