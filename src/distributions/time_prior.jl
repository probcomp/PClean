using CSV

"""
    TimePrior(

Return a random time stamp of form @sprintf("%d:%02d %s", hours, minutes, ampm).

The hours, minutes and ampm are drawn uniformly from {1, .., 12}, {0, .., 59}, and {"a.m.", "p.m."} respectively.
"""
struct TimePrior <: PCleanDistribution end

has_discrete_proposal(::TimePrior) = true

# Assume proposal_atoms are unique.
function discrete_proposal(::TimePrior, proposal_atoms::Vector{String})
  options = [proposal_atoms..., proposal_dummy_value]
  probs = map(x -> isnothing(match(r"^\d?\d:\d\d [ap]\.m\.$", x)) ? -Inf : -log(1440), proposal_atoms)
  total = logsumexp(probs)
  probs = [probs..., log1p(-exp(total))]
  return (options, probs)
end

discrete_proposal_dummy_value(::TimePrior, proposal_atoms::Vector{String}) = begin
  "**:** p.m."
end

random(::TimePrior, proposal_atoms::Vector{String}) = begin
  "$(rand(DiscreteUniform(1, 12))):$(rand(DiscreteUniform(1, 60))) $((rand(Bernoulli(0.5)) == 1) ? "a.m." : "p.m.")"
end

function logdensity(::TimePrior, observed::String, proposal_atoms::Vector{String})
  return -log(1440.0)
end

export TimePrior
