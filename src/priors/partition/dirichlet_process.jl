"""Dirichlet-process partition prior with fixed positive strength `alpha`."""
struct DirichletProcess <: PartitionPrior
    alpha::Float64
    function DirichletProcess(alpha)
        a = finite_real(alpha, "strength")
        a > 0 || throw(ArgumentError("Dirichlet-process strength must be positive"))
        new(a)
    end
end


discount(::DirichletProcess) = 0.0
