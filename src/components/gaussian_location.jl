"""Normal observations with known `observation_sd` and a Normal location base."""
struct GaussianLocation <: ComponentModel
    y::Vector{Float64}
    observation_sd::Float64
    base_mean::Float64
    base_sd::Float64
    function GaussianLocation(y; observation_sd=1.0, base_mean=0.0, base_sd=2.0)
        isempty(y) && throw(ArgumentError("GaussianLocation needs at least one observation"))
        yy = [finite_real(x, "observation") for x in y]
        sigma = finite_real(observation_sd, "observation_sd")
        mu = finite_real(base_mean, "base_mean")
        tau = finite_real(base_sd, "base_sd")
        for (name, sd) in (("observation_sd", sigma), ("base_sd", tau))
            sd > 0 || throw(ArgumentError("$name must be positive"))
            isfinite(sd^2) && sd^2 > 0 && isfinite(inv(sd^2)) ||
                throw(ArgumentError("$name is outside the supported Float64 variance range"))
        end
        isfinite(sigma^2 + tau^2) ||
            throw(ArgumentError("observation and base variances overflow Float64 when added"))
        model = new(yy, sigma, mu, tau)
        all(isfinite(logpdf(Normal(mu, hypot(sigma,tau)), x)) for x in yy) &&
            isfinite(logmarginal(model, eachindex(yy))) ||
            throw(ArgumentError("data and scales give nonfinite Gaussian marginal densities in Float64"))
        posterior_parameters(model, eachindex(yy))
        model
    end
end

loglikelihood(model::GaussianLocation, y, theta) =
    logpdf(Normal(theta, model.observation_sd), y)
logbase(model::GaussianLocation, theta) = logpdf(Normal(model.base_mean, model.base_sd), theta)
randbase(rng::AbstractRNG, model::GaussianLocation) = rand(rng, Normal(model.base_mean, model.base_sd))

# Sequential conjugate updates avoid subtracting large sums of squares.
function posterior_parameters(model::GaussianLocation, indices)
    mean, variance = model.base_mean, model.base_sd^2
    obsvariance = model.observation_sd^2
    for i in indices
        denominator = variance + obsvariance
        mean += (variance/denominator)*(model.y[i]-mean)
        variance *= obsvariance/denominator
    end
    isfinite(mean) && isfinite(variance) && variance > 0 ||
        throw(ArgumentError("Gaussian posterior parameters are outside the supported Float64 range"))
    mean, sqrt(variance)
end

function logmarginal(model::GaussianLocation, indices)
    mean, variance = model.base_mean, model.base_sd^2
    obsvariance = model.observation_sd^2
    value = 0.0
    for i in indices
        denominator = variance + obsvariance
        value += logpdf(Normal(mean, sqrt(denominator)), model.y[i])
        mean += (variance/denominator)*(model.y[i]-mean)
        variance *= obsvariance/denominator
    end
    value
end

function randposterior(rng::AbstractRNG, model::GaussianLocation, indices)
    mean, sd = posterior_parameters(model, indices)
    rand(rng, Normal(mean, sd))
end

function logpredictive(model::GaussianLocation, indices, i::Integer)
    mean, sd = posterior_parameters(model, indices)
    logpdf(Normal(mean, hypot(sd, model.observation_sd)), model.y[i])
end
