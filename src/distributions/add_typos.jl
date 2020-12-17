import StringDistances: DamerauLevenshtein, evaluate

struct AddTypos <: PCleanDistribution end

has_discrete_proposal(::AddTypos) = false

supports_explicitly_missing_observations(::AddTypos) = true

perform_typo(typo, word) = begin
  #return word
  if typo == :insert
    index = rand(DiscreteUniform(0, length(word)))
    letter = collect('a':'z')[rand(DiscreteUniform(1, 26))]
    return "$(word[1:index])$(letter)$(word[index+1:end])"
  end
  if typo == :delete
    index = rand(DiscreteUniform(1, length(word)))
    return "$(word[1:index-1])$(word[index+1:end])"
  end
  if typo == :substitute
    index = rand(DiscreteUniform(1, length(word)))
    letter = collect('a':'z')[rand(DiscreteUniform(1, 26))]
    return "$(word[1:index-1])$letter$(word[index+1:end])"
  end
  if typo == :transpose
    if length(word) == 1
      return
    end
    index = rand(DiscreteUniform(1, length(word)-1))
    return "$(word[1:index-1])$(word[index+1])$(word[index])$(word[index+2:end])"
  end
end

random(::AddTypos, word::String, max_typos=nothing) = begin
  num_typos = rand(NegativeBinomial(ceil(length(word) / 5.0), 0.9))
  num_typos = isnothing(max_typos) ? num_typos : min(max_typos, num_typos)
  for i=1:num_typos
    # Randomly insert/delete/transpose/substitute
    typo = [:insert, :delete, :transpose, :substitute][rand(DiscreteUniform(1, 4))]
    word = perform_typo(typo, word)
  end
  return word
end

const add_typos_density_dict = Dict{Tuple{String, String}, Float64}()
logdensity(::AddTypos, observed::Union{String,Missing}, word::String, max_typos=nothing) = begin
  if ismissing(observed)
    return 0.0
  end

  get!(add_typos_density_dict, (observed, word)) do
    num_typos = evaluate(DamerauLevenshtein(), observed, word)
    if !isnothing(max_typos) && num_typos > max_typos
      return -100000
    end

    l = logpdf(NegativeBinomial(ceil(length(word) / 5.0), 0.9), num_typos)
    l -= log(length(word)) * num_typos
    l -= log(26) * (num_typos) / 2 # Should this be num_typos / 2 (or something smarter)? Maybe we should actually compute the prob of the most probable typo path.
    l
  end
end

export AddTypos
