import Statistics: mean

struct BernoulliProb <: ParamValue{Float64}
  value::Float64
end
struct CategoricalProbs <: ParamValue{Union{Nothing, AbstractArray{Float64, 1}}}
  value :: Union{Nothing, AbstractArray{Float64, 1}}
end

parameter_defaults[:bernoulli] = (BernoulliProb(0.5),)
parameter_defaults[:choose_proportionally] = (nothing, CategoricalProbs(nothing))


compute_suff_stat(::Type{BernoulliProb}, x, all_params) = x ? 1 : 0
compute_suff_stat(::Type{CategoricalProbs}, x, all_params) = (findfirst(isequal(x), all_params[1]), length(all_params[1]))

function default_value(p::IndexedParam{BernoulliProb})
  isempty(p.values) ? 0.5 : mean(map(x -> x.value), values(p.values))
end

function estimator(p::DBModelParam{BernoulliProb})
  mean
end

function default_value(p::IndexedParam{CategoricalProbs})
  nothing
end

function estimator(p::DBModelParam{CategoricalProbs})
  function estimate(values)
    n = values[1][2]
    vs  = map(v -> v[1], values)
    [count(isequal(i), vs) for i=1:n]
  end
end
