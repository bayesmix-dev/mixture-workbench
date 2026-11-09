function canonical_labels(c)
    map=Dict{Int,Int}(); out=similar(c)
    for i in eachindex(c); out[i]=get!(map,c[i],length(map)+1); end
    out
end
function collapsed_logtarget(state)
    s=core(state); m=s.model; M=length(s.S); k=length(s.active); n=length(s.c)
    logp=m.spectrum.logp0-log(-expm1(m.spectrum.logp0))
    for j in s.S; logp+=m.spectrum.logodds[m.spectrum.shell[j]]; end
    logp+=loggamma(m.a*M)-loggamma(n+m.a*M)+s.ld-k*m.logvolume
    for h in s.active
        c=s.clusters[h]
        logp+=loggamma(c.n+m.a)-loggamma(m.a)+log_marginal(c,m)
    end
    logp
end
choose2(n)=n*(n-1)/2
function ari(c,truth)
    counts=Dict{Tuple{Int,Int},Int}(); rows=Dict{Int,Int}(); cols=Dict{Int,Int}()
    for i in eachindex(c)
        pair=(c[i],truth[i]); counts[pair]=get(counts,pair,0)+1
        rows[c[i]]=get(rows,c[i],0)+1; cols[truth[i]]=get(cols,truth[i],0)+1
    end
    a=sum(choose2,values(counts)); b=sum(choose2,values(rows)); d=sum(choose2,values(cols)); total=choose2(length(c))
    expected=b*d/total; denom=(b+d)/2-expected
    denom==0 ? 1.0 : (a-expected)/denom
end
scalar_names()=["k","M","entropy","collapsed_logtarget","k_eq_3","pair_1_2","pair_1_middle","pair_middle_last","coclustering_fraction","ARI"]
function scalar_summary(state)
    s=core(state); n=length(s.c); k=length(s.active)
    entropy=-sum((s.clusters[h].n/n)*log(s.clusters[h].n/n) for h in s.active)
    co=sum(choose2(s.clusters[h].n) for h in s.active)/choose2(n)
    mid=cld(n,2)
    Float64[k,length(s.S),entropy,collapsed_logtarget(s),k==3,s.c[1]==s.c[2],s.c[1]==s.c[mid],s.c[mid]==s.c[end],co,ari(s.c,s.model.truth)]
end
