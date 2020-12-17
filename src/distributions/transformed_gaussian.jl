using Distributions

struct TransformedGaussian <: PCleanDistribution end

struct Transformation
  forward  :: Function
  backward :: Function
  deriv :: Function # x -> |g'(x)|
end

has_discrete_proposal(::TransformedGaussian) = false

random(::TransformedGaussian, mean::Float64, std::Float64, t::Transformation) = t.forward(rand(Normal(mean, std)))

logdensity(::TransformedGaussian, observed::Float64, mean::Float64, std::Float64, t::Transformation) =
  logpdf(Normal(mean, std), t.backward(observed)) - log(abs(t.deriv(t.backward(observed))))


################
# Learned Mean #
################
random(a::TransformedGaussian, mean::MeanParameter, std::Float64, t::Transformation) = random(a, param_value(mean), std, t)
logdensity(a::TransformedGaussian, observed::Float64, mean::MeanParameter, std::Float64, t::Transformation) = logdensity(a, observed, param_value(mean), std, t)

# Incorporating new observations
function incorporate_choice!(::TransformedGaussian, observed::Float64, mean::MeanParameter, std::Float64, t::Transformation)
  incorporate_choice!(AddNoise(), t.backward(observed), mean, std)
end

# Incorporating new observations
function unincorporate_choice!(::TransformedGaussian, observed::Float64, mean::MeanParameter, std::Float64, t::Transformation)
  unincorporate_choice!(AddNoise(), t.backward(observed), mean, std)
end

export TransformedGaussian, Transformation
