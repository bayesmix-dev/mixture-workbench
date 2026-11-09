"""
Source-level component interface. Concrete models provide observations and
the ordinary Julia methods below. A collapsed sweep additionally requires
`logmarginal`; samplers that retain component parameters use the base draws,
likelihood and posterior-update methods. Missing methods fail explicitly.
"""
abstract type ComponentModel end
observations(model::ComponentModel) = model.y
loglikelihood(model::ComponentModel, y, theta) =
    throw(ArgumentError("implement loglikelihood(model, y, theta) for $(typeof(model))"))
logbase(model::ComponentModel, theta) =
    throw(ArgumentError("implement logbase(model, theta) for $(typeof(model))"))
randbase(rng::AbstractRNG, model::ComponentModel) =
    throw(ArgumentError("implement randbase(rng, model) for $(typeof(model))"))
logmarginal(model::ComponentModel, indices) =
    throw(ArgumentError("collapsed sampling requires logmarginal(model, indices) for $(typeof(model))"))
randposterior(rng::AbstractRNG, model::ComponentModel, indices) =
    throw(ArgumentError("implement randposterior(rng, model, indices) for $(typeof(model))"))
logpredictive(model::ComponentModel, indices, i::Integer) =
    logmarginal(model, vcat(collect(indices), i)) - logmarginal(model, indices)
