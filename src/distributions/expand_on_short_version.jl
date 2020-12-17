struct ExpandOnShortVersion <: PCleanDistribution end

supports_explicitly_missing_observations(::ExpandOnShortVersion) = true


function is_short_version(short, long)
  a, b = 1, 1
  s, l = length(short), length(long)
  while a <= s && b <= l
    if lowercase(short[a]) == lowercase(long[b])
      a += 1
    end
    b += 1
  end
  if a > s
    return true
  end
  return false
end


function random(::ExpandOnShortVersion, val, options)
  options = [x for x in options if is_short_version(val, x)]
  if isempty(options)
    return val
  end
  options[rand(Distributions.DiscreteUniform(1, length(options)))]
end

function logdensity(::ExpandOnShortVersion, observed, val, options)
  if ismissing(observed) && in(val, options)
    return 0.0
  elseif ismissing(observed)
    return -1000.0
  end

  if is_short_version(val, observed)
    return -log(length(filter(x -> is_short_version(val, x), options)))
  end
  return -1000.0
end

has_discrete_proposal(::ExpandOnShortVersion) = false
export ExpandOnShortVersion
