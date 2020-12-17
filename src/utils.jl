"""
    normalize(proportions::AbstractArray{T}) where {T <: Real}

Given a vector of non-negative numbers with a positive sum,
return a vector of probabilities that sums to 1.
"""
function normalize(proportions::AbstractArray{T}) where {T <: Real}
  proportions ./ sum(proportions)
end

"""
    logsumexp(logits::AbstractArray{T}) where {T <: Real}

Sum a vector of numbers in log-space.
"""
function logsumexp(logits::AbstractArray{T}) where {T <: Real}
  isempty(logits) && return -Inf
  max_logit = maximum(logits)
  max_logit == -Inf ? -Inf : max_logit + log(sum(exp.(logits .- max_logit)))
end

function logsumexp(x1::Real, x2::Real)
  m = max(x1, x2)
  return m == -Inf ? m : m + log(exp(x1 - m) + exp(x2 - m))
end


"""
    logprobs(proportions::AbstractArray{T}) where {T <: Real}

Like `normalize`, but in log-space and more numerically stable.
"""
logprobs(proportions::Vector{Float64}) = begin
  l = log.(proportions)
  #l .- logsumexp(l)
end


function remove_missing(v::Vector{Union{Missing, T}}) where T
  T[filter(x -> !ismissing(x), v)...]
end
function remove_missing(v::Vector{Missing})
  []
end
function remove_missing(v::Vector{T}) where T
  v
end

export remove_missing
