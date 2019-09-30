import Statistics: mean, stdm

struct NormalMean <: ParamValue{Float64}
  value :: Float64
end

struct NormalStd <: ParamValue{Float64}
  value :: Float64
end

parameter_defaults[:normal] = (NormalMean(0), NormalStd(1))

compute_suff_stat(::Type{NormalMean}, x, all_params) = x
compute_suff_stat(::Type{NormalStd}, x, all_params) = x - all_params[1]


function default_value(p::IndexedParam{NormalMean})
  isempty(p.values) ? 0 : mean(map(x -> x.value, values(p.values)))
end

function estimator(p::DBModelParam{NormalMean})
  mean
end

function default_value(p::IndexedParam{NormalStd})
  isempty(p.values) ? 1.0 : mean(map(x -> x.value, values(p.values)))
end

function estimator(p::DBModelParam{NormalStd})
  function estimate(values)
    stdm(values, 0.0; corrected = false)
  end
end
