struct FormatName <: PCleanDistribution end

supports_explicitly_missing_observations(::FormatName) = true

function random(::FormatName, first, middle, last)
  if ismissing(middle) || middle == "" || rand(Bernoulli(0.1))
    return (rand(Bernoulli(0.1))) ? "$(first[1]). $(last)" : "$(first) $(last)"
  end
  first = (rand(Bernoulli(0.1)))  ? "$(first[1])." : first
  middle = (rand(Bernoulli(0.1))) ? "$(middle[1])." : middle
  return "$(first) $(middle) $(last)"
end

function logdensity(::FormatName, observed, first, middle, last)
  if ismissing(observed)
    return 0.0
  end
  if lowercase(observed) == lowercase("$(first) $(middle) $(last)")
    return log(0.9) + log(0.9) + log(0.9)
  elseif lowercase(observed) == lowercase("$(first) $(last)")
    return log(0.1)
  # For now don't model initials
  else
    return -1000
  end
end


function random(::FormatName, name)
  return length(name) == 0 || rand(Bernoulli(0.5)) ? name : "$(name[1])."
end

function logdensity(::FormatName, observed, name)
  if ismissing(observed)
    if ismissing(name) || name == ""
      return 0.0
    elseif occursin("*", name)
      return -1000.0
    else
      return -5.0
    end
#    return !ismissing(name) && occursin("*", name) ? -1000.0 : 0.0 # this is a silly hack
  end
  if name == ""
    return -1000
  end
  if lowercase(observed) == lowercase(name)
    return log(0.9999)
  elseif lowercase(observed) == lowercase("$(name[1]).")
    return log(0.0001)
  # For now don't model initials
  else
    return -1000
  end
end




# Doesn't need proposal
has_discrete_proposal(::FormatName) = false

# function discrete_proposal(::MaybeSwap, val, options, prob::T) where T <: Real
#   ([val, options...], [log1p(-prob), fill(log(prob) - log(length(options)), length(options))...])
# end


export FormatName
