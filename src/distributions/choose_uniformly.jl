struct ChooseUniformly <: PCleanDistribution end

function random(::ChooseUniformly, options)
  options[rand(Distributions.DiscreteUniform(1, length(options)))]
end

function logdensity(::ChooseUniformly, observed, options)
  # Assume it's possible
  return -log(length(options))
end

has_discrete_proposal(::ChooseUniformly) = true

function discrete_proposal(::ChooseUniformly, options)
  (options, fill(-log(length(options)), length(options)))
end

export ChooseUniformly
