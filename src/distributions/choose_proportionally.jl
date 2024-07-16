"""
  struct ChooseProportionally <: PCleanDistribution
  
A categorical distribution.
"""
struct ChooseProportionally <: PCleanDistribution end

function random(::ChooseProportionally, options, probs::AbstractArray{T}) where T <: Real
  options[rand(Distributions.Categorical(normalize(probs)))]
end

function logdensity(::ChooseProportionally, observed, options, probs::AbstractArray{T}) where T <: Real
  relevant_logprobs = logprobs(probs)[options .== observed]
  isempty(relevant_logprobs) && return -Inf
  logsumexp(relevant_logprobs)
end

has_discrete_proposal(::ChooseProportionally) = true

function discrete_proposal(::ChooseProportionally, options, probs::AbstractArray{T}) where T <: Real
  (options, logprobs(probs))
end


#######################
# Learned Proportions #
#######################
struct VariableSizeProportionsParameterPrior <: ParameterPrior
  concentration :: Float64
end

struct ProportionsParameterPrior
  concentrations :: Vector{Float64}
end

struct ProportionsParameter <: BasicParameter
  current_value :: Vector{Float64}
  prior :: Union{ProportionsParameterPrior, VariableSizeProportionsParameterPrior}
  sample_counts :: Vector{Int}
end

default_prior(::Type{ProportionsParameter}) = VariableSizeProportionsParameterPrior(1.0)
default_prior(::Type{ProportionsParameter}, concentrations::Vector{Float64}) = ProportionsParameterPrior(concentrations)
default_prior(::Type{ProportionsParameter}, num_options::Int) = ProportionsParameterPrior(ones(Float64, num_options))
default_prior(::Type{ProportionsParameter}, concentration::Float64) = VariableSizeProportionsParameterPrior(concentration)
dirichlet_concentrations(prior::VariableSizeProportionsParameterPrior, num_options::Int) = prior.concentration .* ones(Float64, num_options)
dirichlet_concentrations(prior::ProportionsParameterPrior, num_options::Int) = prior.concentrations

function initialize_parameter(::Type{ProportionsParameter}, prior)
  ProportionsParameter(Float64[], prior, Int[])
end

function param_value(p::ProportionsParameter, options)
  if isempty(p.current_value)
    num_options = length(options)
    copy!(p.sample_counts, zeros(Int, num_options))
    copy!(p.current_value, rand(Dirichlet(dirichlet_concentrations(p.prior, num_options))))
  end
  p.current_value
end

function incorporate_choice!(::ChooseProportionally, observed, options, p::ProportionsParameter)
  idx = findfirst(x -> x == observed, options)
  p.sample_counts[idx] += 1
end

function unincorporate_choice!(::ChooseProportionally, observed, options, p::ProportionsParameter)
  idx = findfirst(x -> x == observed, options)
  if p.sample_counts[idx] <= 0
    println(observed, p.sample_counts)
  end
  p.sample_counts[idx] -= 1
end

function resample_value!(p::ProportionsParameter)
  num_options = length(p.current_value)
  dirichlet_prior = dirichlet_concentrations(p.prior, num_options)
  copy!(p.current_value, rand(Dirichlet(dirichlet_prior .+ p.sample_counts)))
end

discrete_proposal(c::ChooseProportionally, options, probs::ProportionsParameter) = discrete_proposal(c, options, param_value(probs, options))
random(c::ChooseProportionally, options, probs::ProportionsParameter) = random(c, options, param_value(probs, options))
logdensity(c::ChooseProportionally, observed, options, probs::ProportionsParameter) = logdensity(c, observed, options, param_value(probs, options))

export ChooseProportionally
