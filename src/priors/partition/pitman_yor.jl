"""
Pitman–Yor partition prior with fixed strength `alpha` and discount `d`.
The support is `0 <= d < 1` and `alpha > -d`; at `d == 0`, `alpha > 0`.
This includes negative strength when `d > 0`.
"""
struct PitmanYor <: PartitionPrior
    alpha::Float64
    d::Float64
    function PitmanYor(alpha, d)
        a = finite_real(alpha, "strength")
        dd = finite_real(d, "discount")
        0 <= dd < 1 || throw(ArgumentError("discount must satisfy 0 <= discount < 1"))
        a > -dd || throw(ArgumentError("strength must exceed -discount (and be positive at discount zero)"))
        new(a, dd)
    end
end


discount(p::PitmanYor) = p.d
