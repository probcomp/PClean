abstract type PCleanDistribution end

function random end

function logdensity end

struct ProposalDummyValue end
const proposal_dummy_value = ProposalDummyValue()

# Can this distribution enumerate values on which
# its posterior support is concentrated? If the distribution's
# support is not finite, then this proposal also places mass
# on a value of type ProposalDummyValue.
function has_discrete_proposal end

function discrete_proposal end

function discrete_proposal_dummy_value end

supports_explicitly_missing_observations(::PCleanDistribution) = false


# function is_enumerable end

# function enumerate_options end

abstract type Parameter end
abstract type BasicParameter <: Parameter end
abstract type ParameterPrior end

function default_prior end

function initialize_parameter end

function incorporate_choice!(args...)
  @assert all(x -> !(x isa Parameter), args)
end

function unincorporate_choice!(args...)
  @assert all(x -> !(x isa Parameter), args)
end

function resample_value! end

struct IndexedParameter{T <: ParameterPrior, U <: BasicParameter} <: Parameter
  shared_prior :: T
  parameters   :: Dict{Any, U}
end

function Base.getindex(p :: IndexedParameter{T, U}, idx) where {T, U}
  if haskey(p.parameters, idx)
    return p.parameters[idx]
  end
  p.parameters[idx] = initialize_parameter(U, p.shared_prior)
end

function resample_value!(p::IndexedParameter)
  for (idx, parameter) in p.parameters
    resample_value!(parameter)
  end
end

include("add_noise.jl")
include("add_typos.jl")
include("choose_proportionally.jl")
include("string_prior.jl")
include("maybe_swap.jl")
include("choose_uniformly.jl")
include("number_code_prior.jl")
include("unmodeled.jl")
include("transformed_gaussian.jl")
include("time_prior.jl")
include("format_name.jl")
include("expand_on_short_version.jl")
