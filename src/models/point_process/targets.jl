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
