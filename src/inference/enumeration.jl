# Supernodes store their "original" ID.
# vmap also maps my values into supermodel IDs NOT stored as supernodes (because they are the submodel nodes of the supermodel.)
# vmap is about *me* and *my* randomchoice, julia, and submodel nodes as they appear in some other trace.

# Now to understand: why levels, instead of flattening.
# It seems it's because we store parent rows per class, instead of
# per *path*. But why?

# Next I have to look at the trace data structure and dependency tracking.

# Also want to remind myself re: submodel nodes.
# Unlike supermodel nodes, submodel nodes actually do 
# have indices changed.
# The thing is we don't represent the whole supermodel, unlike the submodel, which we do represent.



# TODO: don't plan to include nodes that cannot reach any observed nodes.
strip_subnodes(node::SubmodelNode) = strip_subnodes(node.subnode)
strip_subnodes(node) = node

function execute_plan!(block_plan, state::ProposalRowState)
  # Check if any argument in a list of IDs is unavailable,
  # preventing the evaluation of some downstream node.
  function any_unavailable(arg_node_ids)
    any(k -> !haskey(state, k), arg_node_ids)
  end

  # We should never need to process ParameterNodes, LearnedTableNodes, or SupermodelNodes.
  can_process(node::JuliaNode, idx) = !any_unavailable(node.arg_node_ids)
  can_process(node::RandomChoiceNode, idx) = !any_unavailable(node.arg_node_ids) && (haskey(state, idx) || has_discrete_proposal(node.dist))
  can_process(node::ForeignKeyNode, idx) = haskey(state, node.learned_table_node_id)
  can_process(node::SubmodelNode, idx) = haskey(state, idx) || (haskey(state, node.foreign_key_node_id) && can_process(node.subnode, idx))

  # Processing a Julia node.
  function process_node!(node::JuliaNode, idx, branches)
    if any_unavailable(node.arg_node_ids)
      return branch_and_recombine!(branches)
    end

    evaluated_args = [state[arg_id] for arg_id in node.arg_node_ids]
    node_value = node.f(evaluated_args...)
    state[idx] = node_value
    traces = branch_and_recombine!(branches)
    delete!(state, idx)
    return traces
  end

  # Processing a random choice node.
  function process_node!(node::RandomChoiceNode, idx, branches)
    if any_unavailable(node.arg_node_ids) || (!haskey(state, idx) && !has_discrete_proposal(node.dist))
      return branch_and_recombine!(branches)
    end
    # println("Evaluating random choice node $idx")
    evaluated_args = [state[arg] for arg in node.arg_node_ids]
    if haskey(state, idx)
      # Node is observed
      value = state[idx]
      incr_w = logdensity(node.dist, value, evaluated_args...)
      if isinf(incr_w)
        return EmptyTraceCollection()
      end

      traces = branch_and_recombine!(branches)
      # TODO: Do we need to store Dict(idx => value) here, or could this just be empty?
      return product_of_trace_collections(SingletonTraceCollection(Dict(idx => value), incr_w), traces)
    end

    # Node is unobserved and enumerable.
    options, lprobs = discrete_proposal(node.dist, evaluated_args...)
    u = UnionTraceCollection(idx)
    for (option, lprob) in zip(options, lprobs)
      if option isa ProposalDummyValue
        state[idx] = discrete_proposal_dummy_value(node.dist, evaluated_args...)
      else
        state[idx] = option
      end
      traces = branch_and_recombine!(branches)

      # TODO: according to profiling, this is apparently somewhat slow?
      push!(u, option, product_of_trace_collections(SingletonTraceCollection(Dict(idx => option), lprob), traces))
    end
    delete!(state, idx)
    return u
  end

  # We are going to need a couple new kinds of node: a ForeignKeyNode
  # and a SubmodelNode
  function process_node!(node::ForeignKeyNode, idx, branches)
    # TODO: At the moment, this @assert is true.
    #       But if we add HDP-style features, it won't be, so this node
    #       should support observation. (In a Gibbs update, we may
    #       want to consider updating a value that determines how
    #       likely a certain choice is at this node. Currently, this
    #       isn't possible, since this choice only depends on the number of
    #       other customers at each table--and we exploit exchangeabiliy
    #       when updating other customers.)
    @assert !haskey(state, idx) # impossible to observe foreign key...

    # TODO: I believe this should always be true, but in the future,
    # we may need to add a fall-through case for when other args to this
    # node are unavailable. (Currently, there are no other args.)
    @assert haskey(state, node.learned_table_node_id)

    learned_table = state[node.learned_table_node_id]

    # Enumerate existing options in the child table
    existing_row_traces = Dict{Symbol, EnumeratedTraceCollection}()
    # existing_lprobs, new_lprob = pitman_yor_prior_logprobs(learned_table)

    # Consider a limited set of keys
    if !isempty(learned_table.model.hash_keys) && all(x -> haskey(state, node.vmap[x]), learned_table.model.hash_keys)
      # We assume that the values to be hashed on are observed...  TODO: relax this.

      hash_key = [state[node.vmap[i]] for i in learned_table.model.hash_keys]
      if haskey(learned_table.hashed_keys, hash_key)
        valid_key_set = learned_table.hashed_keys[hash_key]
      else
        valid_key_set = Set()
      end
    else
      valid_key_set = keys(learned_table.rows)
    end
    total_referents = learned_table.total_references[]
    log_denominator = log(total_referents + learned_table.pitman_yor_params.strength)
    new_row_lprob = log(learned_table.pitman_yor_params.strength + learned_table.pitman_yor_params.discount * length(learned_table.rows)) - log_denominator
    for foreign_row_key in valid_key_set
      state[idx] = foreign_row_key
      traces = branch_and_recombine!(branches)
      if !isinf(traces.weight)
        # Note: do we need to update sampling/scoring code to handle impossible retained?
        prior_prob = log(learned_table.reference_counts[foreign_row_key] - learned_table.pitman_yor_params.discount) - log_denominator
        traces = product_of_trace_collections(SingletonTraceCollection(Dict(idx => foreign_row_key), prior_prob), traces)
        existing_row_traces[foreign_row_key] = traces
      end
    end

    # Consider adding a new row to child table
    state[idx] = pclean_gensym!("row")
    #new_row_traces = SingletonTraceCollection(Dict(), -500)
    new_row_traces = branch_and_recombine!(branches)
    new_row_trace_collection = product_of_trace_collections(SingletonTraceCollection(Dict(idx => state[idx]), new_row_lprob), new_row_traces)
    # Clean up
    delete!(state, idx)

    # Normalize weights
    total_weight = logsumexp(Float64[t.weight for t in values(existing_row_traces)])
    total_weight = logsumexp(total_weight, new_row_trace_collection.weight)
    return ForeignKeyTraceCollection(idx, existing_row_traces, new_row_trace_collection, total_weight)
  end

  # The only way to get to a Submodel node is if we're considering the implications
  # of -- at *this* level -- a change in foreign key.
  function process_node!(node::SubmodelNode, idx, branches)
    subnode = node.subnode

    # If no key is available, skip this node.
    if !haskey(state, node.foreign_key_node_id)
      return branch_and_recombine!(branches)
    end


    foreign_key = state[node.foreign_key_node_id]

    # In general, we can't assume that the reference indices in `node`
    # are valid indices into state.model.nodes, because we might be evaluating
    # a supermodel's trace. But this is never the case in SubmodelNode, because
    # SupermodelNode{SubmodelNode} is impossible.
    stripped_subnode = strip_subnodes(state.model.nodes[node.foreign_key_node_id])
    @assert stripped_subnode isa ForeignKeyNode
    learned_table = state[stripped_subnode.learned_table_node_id]

    # Case 1: the foreign key is not in the table, i.e., we are generating blind.
    # In that case, just process subnode.
    if !haskey(learned_table.rows, foreign_key)
      #println("Will process submodel node for new subindex ($foreign_key): $idx")
      return process_node!(subnode, idx, branches)
    end

    # Case 2: the foreign key is in the table, and this node is observed.
    #   * A) If the observed value and recorded value are not equal, return an EmptyTraceCollection.
    #   * B) Otherwise, just continue -- the node is observed and will always be observed.
    chosen_row   = learned_table.rows[foreign_key]
    chosen_value = chosen_row[node.subnode_id]
    if haskey(state, idx) # && !ismissing(state[idx]) && !ismissing(chosen_value)
      observed_value = state[idx]
      close_enough = ismissing(observed_value) && ismissing(chosen_value) || (chosen_value isa Real && isapprox(chosen_value, observed_value)) || (!ismissing(chosen_value) && !ismissing(observed_value) && chosen_value == observed_value)
      if !close_enough
        return EmptyTraceCollection()
      end
      return branch_and_recombine!(branches)
    end

    # Case 3: the foreign key is in the table, and this node is not observed.
    #   * If this node would be empty without knowing all the values, then leave it empty, otherwise fill it in.
    #   * (Copy deterministically from the learned table and continue enumeration.)
    if can_process(subnode, idx)
      state[idx] = chosen_value
      traces = branch_and_recombine!(branches)
      delete!(state, idx)
      # May be OK to just return `traces`.
      # Depends on whether the  full proposal tries to avoid
      # recomputing things. But I think it doesn't.
      return traces # product_of_trace_collections(SingletonTraceCollection(Dict(idx => chosen_value), 0.0), traces)
    end
    return branch_and_recombine!(branches)
  end

  function process_node!(node::SupermodelNode, idx, branches)
    # If we are already evaluating a particular parent trace,
    # then we process our supernode, using its index in the original trace.
    if !isnothing(state.active_parent_trace)
      return process_node!(node.supernode, node.supernode_id, branches)
    end

    # Otherwise, we need to loop through the possible traces at node.level.
    superweight = 0.0
  #  println(state.parent_rows)
  #  println(length(state.parent_tables))
  #  println(node.level)
    for parent_row in state.parent_rows[node.level]
      state.active_parent_trace = nothing
      state.parent_trace_recomputed = Dict()

      active_parent_trace = state.parent_tables[node.level].rows[parent_row]

      # Fill choices of parent_trace_recomputed based on levels
      # TODO: improve performance here? or at least profile to see if this is slow.
      for (fkid, vmap) in state.model.tracked_parents[node.level]
        # If it refers to me
        if active_parent_trace[fkid] == state.row_key
          # Fill in the relevant supernode ids  in the state
          for (i, j) in vmap
            # Might be nothing because we haven't gotten there yet.
            state.parent_trace_recomputed[j] = haskey(state, i) ? state[i] : nothing
          end
        end
      end

      state.active_parent_trace = active_parent_trace
      traces = process_node!(node.supernode, node.supernode_id, branches)
      superweight += traces.weight
    end
    state.active_parent_trace = nothing
    state.parent_trace_recomputed = nothing
    return SingletonTraceCollection(Dict(), superweight)
  end

  function branch_and_recombine!(branches)
    processed = EnumeratedTraceCollection[process_node!(state.model.nodes[branch[1]], branch[1], branch[2]) for branch in branches]
    if length(processed) == 1
      return processed[1]
    else
      return cross_product_trace_collection(processed)
    end
  end
  return branch_and_recombine!(block_plan)
end
