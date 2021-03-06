"""
    MaybeSwap(val, options, prob)

With probability `prob`, return a random element from `options`, otherwise return `val`.
"""
struct MaybeSwap <: PCleanDistribution end

supports_explicitly_missing_observations(::MaybeSwap) = true

function random(::MaybeSwap, val, options, prob::T) where T <: Real
  if rand(Bernoulli(prob))
    options[rand(Distributions.DiscreteUniform(1, length(options)))]
  else
    val
  end
end

function logdensity(::MaybeSwap, observed, val, options, prob::T) where T <: Real
  # We treat missing as an observed value.
  # Maybe we need to have a thing that says,
  # if you're missing, then val needs to be
  # one of `options.`
  if ismissing(observed) && in(val, options)
    return 0.0
  elseif ismissing(observed)
    return -1000.0
  end

  if val == observed
    return log1p(-prob)
  end
  return log(prob) - log(length(options))
end

has_discrete_proposal(::MaybeSwap) = false

# function discrete_proposal(::MaybeSwap, val, options, prob::T) where T <: Real
#   ([val, options...], [log1p(-prob), fill(log(prob) - log(length(options)), length(options))...])
# end



#######################
# Learned Error Prob #
#######################
struct ProbParameterPrior <: ParameterPrior
  a :: Float64
  b :: Float64
end

mutable struct ProbParameter <: BasicParameter
  current_value :: Float64
  prior :: ProbParameterPrior
  heads :: Int
  tails :: Int
end

default_prior(::Type{ProbParameter}) = ProbParameterPrior(1.0, 3.0)
default_prior(::Type{ProbParameter}, odds::Float64) = ProbParameterPrior(odds * 4, (1 - odds) * 4)
default_prior(::Type{ProbParameter}, a::Float64, b::Float64) = ProbParameterPrior(a, b)

function initialize_parameter(::Type{ProbParameter}, prior)
  ProbParameter(rand(Beta(prior.a, prior.b)), prior, 0, 0)
end

function param_value(p::ProbParameter)
  p.current_value
end

function incorporate_choice!(::MaybeSwap, observed, val, options, p::ProbParameter)
  if ismissing(observed)
    return
  end
  if observed == val
    p.tails += 1
  else
    p.heads += 1
  end
end

function unincorporate_choice!(::MaybeSwap, observed, val, options, p::ProbParameter)
  if ismissing(observed)
    return
  end
  if observed == val
    p.tails -= 1
  else
    p.heads -= 1
  end
end

function resample_value!(p::ProbParameter)
  p.current_value = rand(Beta(p.prior.a + p.heads, p.prior.b + p.tails))
end

random(c::MaybeSwap, val, options, prob::ProbParameter) = random(c, val, options, param_value(prob))
logdensity(c::MaybeSwap, observed, val, options, prob::ProbParameter) = logdensity(c, observed, val, options, param_value(prob))






export MaybeSwap
