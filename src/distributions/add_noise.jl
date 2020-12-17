struct AddNoise <: PCleanDistribution end

has_discrete_proposal(::AddNoise) = false

random(::AddNoise, mean::Float64, std::Float64) = rand(Normal(mean, std))

logdensity(::AddNoise, observed::Float64, mean::Float64, std::Float64) = logpdf(Normal(mean, std), observed)


################
# Learned Mean #
################

# Normal prior for the mean of a normal distribution
struct MeanParameterPrior <: ParameterPrior
  mean :: Float64
  std  :: Float64
end

# Parameter representing the mean of a normal distribution
mutable struct MeanParameter <: BasicParameter
  current_value  :: Float64
  prior          :: MeanParameterPrior
  sample_counts  :: Vector{Int}
  sample_sums    :: Vector{Float64}
  sample_stds    :: Vector{Float64}
end

# User-specified priors
default_prior(::Type{MeanParameter}) = begin
  @error "Please specify a reasonable default for the mean parameter."
end
default_prior(::Type{MeanParameter}, mean) = MeanParameterPrior(mean, 0.5 * abs(mean))
default_prior(::Type{MeanParameter}, mean, std) = MeanParameterPrior(mean, std)


# Looking up the parameter's value
param_value(p::MeanParameter) = p.current_value
random(a::AddNoise, mean::MeanParameter, std::Float64) = random(a, param_value(mean), std)
logdensity(a::AddNoise, observed::Float64, mean::MeanParameter, std::Float64) = logdensity(a, observed, param_value(mean), std)

# Initialization
function initialize_parameter(::Type{MeanParameter}, prior::MeanParameterPrior)
  MeanParameter(rand(Normal(prior.mean, prior.std)), prior, Int[], Float64[], Float64[])
end

# Incorporating new observations
function incorporate_choice!(::AddNoise, observed::Float64, mean::MeanParameter, std::Float64)
  idx = findfirst(x -> isapprox(x, std), mean.sample_stds)
  if isnothing(idx)
    push!(mean.sample_stds, std)
    push!(mean.sample_sums, observed)
    push!(mean.sample_counts, 1)
    return
  end
  mean.sample_counts[idx] += 1
  mean.sample_sums[idx] += observed
end

# Unincorporating new observations
function unincorporate_choice!(::AddNoise, observed::Float64, mean::MeanParameter, std::Float64)
  idx = findfirst(x -> isapprox(x, std), mean.sample_stds)
  @assert !isnothing(idx)
  mean.sample_counts[idx] -= 1
  mean.sample_sums[idx] -= observed
  if iszero(mean.sample_counts[idx])
    deleteat!(mean.sample_counts, idx)
    deleteat!(mean.sample_sums, idx)
    deleteat!(mean.sample_stds, idx)
  end
end

# Gibbs update
function resample_value!(m::MeanParameter)
  mean, var = m.prior.mean, m.prior.std^2
  for (count, sum, std) in zip(m.sample_counts, m.sample_sums, m.sample_stds)
    # TODO: is this stable?
    new_var = 1.0 / (1.0/var + count/(std^2))
    mean, var = new_var * (mean/var + sum/std^2), new_var
  end
  m.current_value = rand(Normal(mean, sqrt(var)))
end

export AddNoise
