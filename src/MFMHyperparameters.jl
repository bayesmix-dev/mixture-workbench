module MFMHyperparameters

using Random
import ..MFMBaseline as MF

export slice_gamma!,log_gamma_density

"""Conditional log density on eta=log(gamma), including the Jacobian eta."""
function log_gamma_density(s::MF.State,eta::Real)
    isfinite(eta) || return -Inf
    g=exp(Float64(eta))
    isfinite(g) && g>0 || return -Inf
    value=MF.gamma_logtarget(s,g)+eta
    isnan(value) && error("nonfinite log-gamma conditional evaluation")
    value
end

"""
Exact stepping-out/shrinkage slice update of log(gamma). The target includes the
log-transform Jacobian. Stepping out has no imposed gamma bound: safeguards abort
rather than silently truncating a bracket. The original state and RNG remain
unchanged if a safeguard or numerical evaluation fails.

This optional extra kernel leaves the baseline sampler unchanged. It may be
composed after MFMBaseline.sweep!; the existing gamma MH counters continue to
refer only to that baseline update. Slice diagnostics are returned separately.
"""
function slice_gamma!(s::MF.State;rng::AbstractRNG=s.rng,width=1.,max_steps=100000,max_shrink=100000)
    isfinite(width) && width>0 || throw(ArgumentError("slice width must be positive and finite"))
    max_steps isa Integer && max_steps>=1 || throw(ArgumentError("max_steps must be positive"))
    max_shrink isa Integer && max_shrink>=1 || throw(ArgumentError("max_shrink must be positive"))
    isfinite(s.gamma) && s.gamma>0 || throw(ArgumentError("current gamma must be positive and finite"))
    trialrng=copy(rng)
    hasmethod(copy!,Tuple{typeof(rng),typeof(trialrng)}) || throw(ArgumentError("slice RNG must support copy!"))
    current=log(s.gamma);evaluations=0
    function density(eta)
        evaluations+=1
        log_gamma_density(s,eta)
    end
    atcurrent=density(current);isfinite(atcurrent) || error("invalid current log-gamma target")
    height=atcurrent-randexp(trialrng)
    left=current-width*rand(trialrng);right=left+width;steps=0
    while density(left)>height
        steps+=1;steps<=max_steps || error("gamma slice stepping-out safeguard exhausted")
        left-=width
    end
    while density(right)>height
        steps+=1;steps<=max_steps || error("gamma slice stepping-out safeguard exhausted")
        right+=width
    end
    for shrink in 0:max_shrink-1
        candidate=left+(right-left)*rand(trialrng)
        if density(candidate)>=height
            value=exp(candidate)
            isfinite(value) && value>0 || error("invalid accepted gamma slice draw")
            copy!(rng,trialrng)
            s.gamma=value
            return (gamma=value,evaluations=evaluations,steps=steps,shrinks=shrink)
        end
        candidate<current ? (left=candidate) : (right=candidate)
    end
    error("gamma slice shrinkage safeguard exhausted")
end

end
