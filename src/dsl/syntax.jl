import Gen
using Gen
using MacroTools
import JuliaDB

#  A DBModel consists of a generative function,
#  an alignment, and a vector of parameters.
# The generative function:
#   * accepts parameters as arguments
#   * returns a NamedTuple, the "clean" row.
#     this clean row must _also_ contain any parameters?
#
#   * traces dirty values at addresses indicated
#   *

"""
  struct DBModel

A model of a data table, constructed with `@dbmodel`.
Contains:
  * a Gen generative function
  * an alignment, mapping DB column names to trace addresses
  * a vector of names in the order in which they are sampled
  * a vector of parameters
"""
struct DBModel
  model::GenerativeFunction
  alignment::Dict
  namesInOrder::Vector{Symbol}
  params::Vector{DBModelParam}
end


function parse_dbmodel_expr(expr)
  is_sampling_line = (@capture(expr, @keyword_ var_Symbol = tracedcall_) ||
    @capture(expr, @keyword_ name_ = tracedcall_) ||
    @capture(expr, @keyword_ name_ var_ = tracedcall_)) &&
    (keyword == Symbol("@column") || keyword == Symbol("@dirty"))

  if !is_sampling_line
    return false, nothing
  end

  name = !@isdefined(name) || name === nothing ? string(var) : name
  var  = !@isdefined(var)  || var === nothing  ? gensym(name) : var
  canonname = JuliaDB.canonical_name(name)
  is_dirty = keyword == Symbol("@dirty")

  true, (is_dirty, var, canonname, tracedcall)
end

"""
  @pclean model_name begin ... end

Creates a DBModel object from code.
"""
macro pclean(model_name, ex)
  # A variable name that, inside the GF,
  # will accumulate the names of columns
  # in the returned row.
  column_names_var = gensym("colnames")
  # A variable name that will accumulate the values of columns
  # to return, in the GF.
  column_values_var = gensym("colvals")
  # Auxiliary versions
  aux_cols_names_var = gensym("auxcolnames")
  aux_cols_values_var = gensym("auxcolvals")

  # The name of the argument to the generative function specifying
  # the address at which the function should stop and return.
  stop_here_name = gensym("stopping_point")

  # The name of the generative function we will create
  generative_function_name = gensym(model_name)
  # An alignment from column names to trace addresses
  alignment = Dict()
  # A dictionary taking names to numbers. We will later extract from this an array
  # of names ordered according to these numbers. Represents the order in which
  # observable names are sampled.
  observableOrder = Dict()
  currentObservableN = 1

  # Lists of parameter names and values (i.e., the initial Parameter objects)
  param_names = []
  param_objs = []

  function convert_expression(expr)
    # Try parsing the expression
    success, parsed_vals = parse_dbmodel_expr(expr)

    # If this wasn't a sampling expression (@true, @dirty, ...),
    # don't change it.
    if !success
      return expr
    end

    # Otherwise, extract the necessary information:
    #   is_dirty: was this an @dirty expression?
    #   var: the (Symbol) variable name
    #   name: the (Symbol) column name
    #   tracedcall: the distribution(arg1, ..., argN) expression
    (is_dirty, var, name, tracedcall) = parsed_vals

    # A place to accumulate the expressions we will return
    # (to be wrapped in a begin...end block). These go inside
    # the generative function, replacing the @true/@dirty expression.
    generated_exprs = []

    # Will be a list of param_names.
    # For each, generate
    #  push!(aux_cols_names_var, param_name.column)
    #  push!(aux_cols_vals_var, compute_suff_stat(param_name, column_name, vec)
    needed_suff_stats = []

    # Trace address for this random choice
    addr = gensym(is_dirty ? "dirty_$name" : name)

    # Add to alignment if necessary
    if is_dirty || !haskey(alignment, name)
       alignment[name] = addr
       observableOrder[name] = currentObservableN
       currentObservableN += 1
    end

    # Generated code structure:
    # If no learned parameters, just trace the traced call, and potentially add
    # to the return value.
    # If there are learned parameters, compute all parameters, and pack them into
    # a vector e.g. "Rent_all_params". Add this to the thing.
    # For any indexed parameters, also add the index.
    # Then make the traced call, and add the field.


    # Generated code example for @true "Rent" rent = normal(_[city, state], _)

    # rent_normal_mean_index = (city, state)
    # rent_normal_mean_param = rent_normal_mean_param3[rent_normal_mean_index]


    # rent = @trace(normal(rent_normal_mean_param, rent_normal_std_param), :Rent)
    # push!(colnames, :Rent)
    # push!(colvals, rent)
    # push!(colnames, :rent_normal_mean_index)
    # push!(colvals, rent_normal_mean_index)
    # push!(colnames, :Rent_params)
    # push!(colvals, [rent_normal_mean_param, rent_normal_std_param])

    # By default, only "true" columns
    # should be added to the inferred "clean" table.
    add_to_table = !is_dirty

    # Think about parameters here. For now, we will not use
    # Gen's @param functionality, but rather pass params in
    # as arguments.
    # This means we have to maintain a list of parameters.
    if !@capture(tracedcall, dist_(params__))
      throw("Sampling statement must be of the form distribution(param1, ..., paramN).")
    end

    has_learned_params = any(x -> string(x)[1] == '_', params)
    if has_learned_params
      # add_to_table = true

      # Calculate all params.
      all_params_column_name = gensym("$(name)_all_params")
      for idx=1:length(params)
        # We have three cases: non-learned, simple, indexed.
        # Non-learned: create a new name, assign it to the expression,
        #   replace the expression in tracedcall.
        # Simple: create a new name, add it to the list of names, and
        #   replace in tracedcall.
        # Indexed: create _three_ new names: param_name, param_expr_name, and index_expr_name.
        #   Add param_name to list of param names, assign index_expr_name to the index expression,
        #   add index to aux vars, and assign param_expr_name to the indexing expression.
        #   Replace in tracedcall.
        # In simple and indexed cases both, we also need to make sure that the parameter itself
        # is created and added as a Value. It needs references to the aux vars (other_params, maybe index)

        # Finally, add vector of params (current tracedcall expression) to aux vars.
        param_expr = params[idx]
        params[idx] = gensym("$(name)_param_$idx")

        if param_expr == :_
          new_param_obj = create_simple_parameter(dist, idx, name)
          push!(param_names, params[idx])
          push!(param_objs, new_param_obj)
          push!(needed_suff_stats, params[idx])
          params[idx] = :(PClean.param_value($(params[idx])))
        elseif string(param_expr)[1] == '_' && @capture(param_expr, _[indices__])
          # Compute index expression, add to auxiliary fields
          index_expression_name = gensym("$(name)_param_$(idx)_index")
          if length(indices) > 1
            push!(generated_exprs, :($index_expression_name = ($(indices...),)))
          else
            push!(generated_exprs, :($index_expression_name = $(indices[1])))
          end
          push!(generated_exprs, :(push!($aux_cols_names_var, $(Meta.quot(index_expression_name)))))
          push!(generated_exprs, :(push!($aux_cols_values_var, $index_expression_name)))

          # Add this learned parameter
          new_parameter_name = gensym("$(name)_param_$(idx)_dict")
          new_parameter_obj = create_indexed_parameter(dist,idx,name,index_expression_name)
          push!(param_names, new_parameter_name)
          push!(param_objs, new_parameter_obj)
          push!(needed_suff_stats, new_parameter_name)

          # Compute actual instantiated value of this parameter
          push!(generated_exprs, :($(params[idx]) = $new_parameter_name[$index_expression_name]))
        else
          # This is just a regular parameter
          push!(generated_exprs, :($(params[idx]) = $param_expr))
        end
      end

      tracedcall = :($dist($(params...)))
    end

    push!(generated_exprs, :($var = @trace($tracedcall, $(Meta.quot(addr)))))
    if haskey(alignment, name)
      push!(generated_exprs, :(if $stop_here_name == $(Meta.quot(addr)); return; end))
    end

    for param_name in needed_suff_stats
      push!(generated_exprs, :(push!($aux_cols_names_var, $(param_name).column)))
      push!(generated_exprs, :(push!($aux_cols_values_var, PClean.compute_suff_stat($(param_name), $var, [$(params...)]))))
    end

    if add_to_table
      push!(generated_exprs, :(push!($column_names_var, $(Meta.quot(name)))))
      push!(generated_exprs, :(push!($column_values_var, $var)))
    end

    :(begin $(generated_exprs...) end)
  end

  body = MacroTools.postwalk(convert_expression, ex)

  gen_expr = esc(:(Gen.@gen function $generative_function_name($stop_here_name, $(param_names...))
      $column_names_var = []
      $column_values_var = []
      $aux_cols_names_var = []
      $aux_cols_values_var = []
      $body
      (NamedTuple{Tuple($column_names_var)}($column_values_var),
       NamedTuple{Tuple($aux_cols_names_var)}($aux_cols_values_var))
    end))

  # Compute observable order:
  namesInOrder = map(x -> x[1], sort(collect(observableOrder), by=x -> x[2]))

  model_name = esc(model_name)
  generative_function_name = esc(generative_function_name)
  quote
    $gen_expr
    $model_name = DBModel($generative_function_name, $alignment, $namesInOrder, [$(param_objs...)])
  end
end

export @pclean
