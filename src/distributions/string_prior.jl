using CSV

struct StringPrior <: PCleanDistribution end

letter_probs_file = joinpath(dirname(pathof(PClean)), "distributions", "lmparams", "letter_probabilities.csv")
letter_trans_file = joinpath(dirname(pathof(PClean)), "distributions", "lmparams", "letter_transition_matrix.csv")
const initial_letter_probs = CSV.File(letter_probs_file; header=false) |> CSV.DataFrame! |> Array{Float64}
const english_letter_transitions = CSV.File(letter_trans_file; header=false) |> CSV.DataFrame! |> Matrix{Float64}
const alphabet = [collect('a':'z')..., ' ', '.']
const alphabet_lookup = Dict([l => i for (i, l) in enumerate(alphabet)])

has_discrete_proposal(::StringPrior) = true

# Assume proposal_atoms are unique.
function discrete_proposal(::StringPrior, min_length::Int, max_length::Int, proposal_atoms::Vector{String})::Tuple{Vector{Union{String, ProposalDummyValue}}, Vector{Float64}}
  options = Union{String, ProposalDummyValue}[proposal_atoms..., proposal_dummy_value]
  probs = map(s -> logdensity(StringPrior(), s, min_length, max_length, proposal_atoms), proposal_atoms)
  total = logsumexp(probs)
  probs = Float64[probs..., log1p(-exp(total))]
  return (options, probs)
end

discrete_proposal_dummy_value(::StringPrior, min_length::Int, max_length::Int, proposal_atoms::Vector{String}) = begin
  join(fill("*", Int(floor((min_length + max_length) / 2))))
end

random(::StringPrior, min_length::Int, max_length::Int, proposal_atoms::Vector{String}) = begin
  len = rand(DiscreteUniform(min_length, max_length))
  letters = []
  for i=1:len
    dist = (i == 1) ? vec(initial_letter_probs) : vec(english_letter_transitions[:, letters[end]])
    if !isprobvec(dist)
      dist = normalize(dist)
    end
    push!(letters, rand(Categorical(dist)))
  end
  join([alphabet[letter] for letter in letters])
end

const UNUSUAL_LETTER_PENALTY = 1000
const string_prior_density_dict = Dict{Tuple{String, Int, Int}, Float64}()
function logdensity(::StringPrior, observed::String, min_length::Int, max_length::Int, proposal_atoms::Vector{String})
  get!(string_prior_density_dict, (observed, min_length, max_length)) do
    if length(observed) < min_length || length(observed) > max_length
      return -Inf
    end
    score = -log(max_length-min_length+1)
    if length(observed) == 0
      return score
    end

    prev_letter = nothing
    for letter in observed
      dist = isnothing(prev_letter) ? initial_letter_probs : vec(english_letter_transitions[:, prev_letter])
      prev_letter = haskey(alphabet_lookup, lowercase(letter)) ? alphabet_lookup[lowercase(letter)] : nothing
      score += isnothing(prev_letter) ? -log(28) : max(log(dist[prev_letter]), -UNUSUAL_LETTER_PENALTY)
    end
    score
  end
end

export StringPrior
