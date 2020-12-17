# Right now, this is for _observed_ number codes, like NPI in Physician.

struct NumberCodePrior <: PCleanDistribution end

function random(::NumberCodePrior)
  @debug "WARNING: Sampling a random number code. That's not good!"
  return 0
end

function logdensity(::NumberCodePrior, val::Int)
  # Assume it's possible
  # Penalize big codes (they are harder to generate)
  return -log(val)
end

has_discrete_proposal(::NumberCodePrior) = false

export NumberCodePrior
