"""Canonical-partition joint density with respect to Lebesgue measure on occupied atoms."""
function joint_logtarget(s::State)
    counts = [count(==(h),s.z) for h in eachindex(s.theta)]
    S.logeppf(s.prior, counts) + sum(logbase(s.model,theta) for theta in s.theta) +
        sum(loglikelihood(s.model,s.model.y[i],s.theta[s.z[i]]) for i in eachindex(s.z))
end

"""Atom-integrated target, available for the conjugate Gaussian model only."""
function collapsed_logtarget(s::State)
    _conjugate(s.model) || throw(ArgumentError("no analytic collapsed target is implemented for this model"))
    counts = [count(==(h),s.z) for h in eachindex(s.theta)]
    S.logeppf(s.prior,counts) +
        sum(S.logmarginal(s.model,findall(==(h),s.z)) for h in eachindex(s.theta))
end

function check_state(s::State)
    length(s.z) == length(s.model.y) && !isempty(s.z) ||
        throw(ArgumentError("allocations must cover every observation"))
    all(z->1<=z<=length(s.theta),s.z) && s.z == S.canonical_labels(s.z) &&
        maximum(s.z) == length(s.theta) || throw(ArgumentError("occupied atoms must match canonical allocations"))
    all(isfinite,s.theta) && isfinite(joint_logtarget(s)) ||
        throw(ArgumentError("state has a nonfinite joint target"))
    s.sweeps>=0 && 0<=s.atom_accepts<=s.atom_attempts ||
        throw(ArgumentError("sampler counters are inconsistent"))
    true
end
