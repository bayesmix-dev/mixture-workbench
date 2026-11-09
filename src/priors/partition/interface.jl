abstract type PartitionPrior end

function finite_real(x, name)
    x isa Real && !(x isa Bool) || throw(ArgumentError("$name must be a finite real number"))
    value = Float64(x)
    isfinite(value) || throw(ArgumentError("$name must be a finite real number"))
    value
end
