strength(p::PartitionPrior) = p.alpha

function valid_counts(counts)
    all(n -> n isa Integer && !(n isa Bool) && n > 0, counts) ||
        throw(ArgumentError("occupied block counts must be positive integers"))
    counts
end

"""
Log probability of one specific set partition with the supplied block sizes.
It includes all prior normalising factors; it has no combinatorial factor for
the number of partitions with those sizes. The empty partition has probability
one. Source: Pitman (2006), Section 3.2, Theorem 3.2, equation (3.6), with his
discount alpha and strength theta written here as d and alpha respectively:
https://www.stat.berkeley.edu/~aldous/206-Exch/Papers/pitman_CSP.pdf
"""
function logeppf(prior::PartitionPrior, counts)
    valid_counts(counts)
    isempty(counts) && return 0.0
    n = sum(counts)
    a, d = strength(prior), discount(prior)
    # Products start after the deterministic first customer. In particular,
    # this avoids a log(alpha) term for the valid negative-strength PY case.
    value = sum((log(a + j*d) for j in 1:(length(counts)-1)); init=0.0)
    value -= sum((log(a + j) for j in 1:(n-1)); init=0.0)
    for count in counts
        value += sum((log(j-d) for j in 1:(count-1)); init=0.0)
    end
    value
end

"""
Unnormalised log seating weights for occupied blocks followed by one new
block. For an empty partition there is one deterministic choice with log
weight zero, including when a Pitman–Yor strength is zero or negative.
"""
function seating_logweights(prior::PartitionPrior, counts)
    valid_counts(counts)
    isempty(counts) && return [0.0]
    a, d = strength(prior), discount(prior)
    vcat([log(n-d) for n in counts], log(a + length(counts)*d))
end
