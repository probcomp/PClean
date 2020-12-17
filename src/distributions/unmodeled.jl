struct Unmodeled <: PCleanDistribution end

function random(::Unmodeled)
  @error "Sampling an unmodeled value."
end

function logdensity(::Unmodeled, obs::Any)
  # Assume it's possible
  return 0.0
end

has_discrete_proposal(::Unmodeled) = false

supports_explicitly_missing_observations(::Unmodeled) = true

export Unmodeled
