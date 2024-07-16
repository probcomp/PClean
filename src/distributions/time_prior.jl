using CSV

struct TimePrior <: PCleanDistribution end

has_discrete_proposal(::TimePrior) = true

# Assume proposal_atoms are unique.
function discrete_proposal(::TimePrior, proposal_atoms::Vector{String})
  options = [proposal_atoms..., PROPOSAL_DUMMY_VALUE]
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
