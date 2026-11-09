"""Stable log of the falling factorial (M)_k; impossible counts have zero mass."""
function logfallingfactorial(M::Integer,k::Integer)
    0<=k<=M || return -Inf
    sum(log(Float64(M-j)) for j in 0:k-1;init=0.)
end
