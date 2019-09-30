using JuliaDB

abstract type ParamValue{T} end
abstract type DBModelParam{T <: ParamValue} end


struct SimpleParam{T <: ParamValue} <: DBModelParam{T}
  value :: T
  column :: Symbol
end

struct IndexedParam{T <: ParamValue} <: DBModelParam{T}
  values :: Dict{Any, T}
  column :: Symbol
  index_column :: Symbol
end

const parameter_defaults = Dict()
include("categorical_params.jl")
include("normal_params.jl")

function Base.getindex(p::IndexedParam{T}, idx)::U where T <: ParamValue{U} where U
  haskey(p.values, idx) ? p.values[idx].value : default_value(p)
end

function update_value(x::T, new) where T <: ParamValue
  T(new)
end

function update_value(x::SimpleParam{T}, new) where T <: ParamValue
  SimpleParam(update_value(x.value, new), x.column)
end

function create_simple_parameter(distribution :: Symbol, idx :: Int, column :: Symbol)
  param_val = parameter_defaults[distribution][idx]
  column = gensym("sufficient_statistic_for_$(column)_param_$idx") # include "dirtyness" info?
  SimpleParam(param_val, column)
end

function create_indexed_parameter(distribution :: Symbol, idx :: Int, column::Symbol, index_column :: Symbol)
  param_val = parameter_defaults[distribution][idx]
  column = gensym("sufficent_statustic_for_$(column)_param_$idx") # include dirtyness info?
  IndexedParam{typeof(param_val)}(Dict(), column, index_column)
end

function param_value(p::SimpleParam{T})::U where T <: ParamValue{U} where U
  p.value.value
end

function update_param(p::SimpleParam{T}, complete_data) where T <: ParamValue
  update_value(p, estimator(p)(JuliaDB.select(complete_data, p.column)))
end

function update_param(p::IndexedParam{T}, complete_data) where T <: ParamValue
  new_vals = JuliaDB.groupby(estimator(p), complete_data, p.index_column; select=p.column)

  IndexedParam{T}(Dict(v[1] => T(v[2]) for v in new_vals), p.column, p.index_column)
end

compute_suff_stat(::DBModelParam{T}, x, all_params) where T <: ParamValue = compute_suff_stat(T, x, all_params)
