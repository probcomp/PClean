import Base: invokelatest

function prune_plan(plan::Plan, state::ProposalRowState)
    # Prune nodes that can't contribute to the score.
    new_plan = Plan(Step{Plan}[])
    class_model = state.trace.model.classes[state.class]
    for step in plan.steps
        k, subplan = step.idx, step.rest
        pruned_subplan = prune_plan(subplan, state)
        if !isempty(pruned_subplan.steps)
            push!(new_plan.steps, Step(k, pruned_subplan))
        elseif haskey(state, k)
            push!(new_plan.steps, Step(k, Plan(Step{Plan}[])))
            
            # TODO: Only include SupermodelNodes for paths that point here.
            # (This may require changes to the JIT proposal compiler too.)
        elseif class_model.nodes[k] isa SupermodelNode
            push!(new_plan.steps, Step(k, Plan(Step{Plan}[])))
        end
    end
    return new_plan
end

function propose_non_enumerable!(vertex_order::Vector{VertexID}, state::ProposalRowState)
    p = 0.0
    q_cont = 0.0
    
    retained_trace = state.retained_trace

    class_model = state.trace.model.classes[state.class]
    
    function process_node!(node::JuliaNode, idx)
        # Can't reuse already-computed value if this is a SuperNode, so recompute.
        evaluated_args = [state[arg] for arg in node.arg_node_ids]
        state[idx] = node.f(evaluated_args...)
    end
    
    function process_node!(node::RandomChoiceNode, idx)
        evaluated_args = [state[arg] for arg in node.arg_node_ids]
        
        # TODO: When is the discrete proposal *not* equivalent to the prior?
        if !haskey(state, idx) && has_discrete_proposal(node.dist)
            # A discrete proposal is available, but was not used. Use it now.
            options, lprobs = discrete_proposal(node.dist, evaluated_args...)
            probs = exp.(lprobs .- logsumexp(lprobs))
            if isnothing(retained_trace)
                chosen_index = rand(Categorical(probs))
            else
                chosen_index = findfirst(x -> x == retained_trace[idx], options)
                if isnothing(chosen_index)
                    chosen_index = findfirst(x -> x isa ProposalDummyValue, options)
                end
            end
            state[idx] = options[chosen_index]
            q_cont += lprobs[chosen_index]
        end
        
        if !haskey(state, idx) || state[idx] isa ProposalDummyValue
            # Sample a value, or use the retained one (which would have been sampled here.)
            state[idx] = isnothing(retained_trace) ? random(node.dist, evaluated_args...) : retained_trace[idx]
        else
            # Observed node.
            incr = logdensity(node.dist, state[idx], evaluated_args...)
            p += incr
        end
    end
    
    function process_node!(node::ForeignKeyNode, idx)
        # TODO: double check that it's correct for the retained particle's weight
        # to be p/q, even though it was not sampled from (this) q (because underlying 
        # parameters may have changed.)
        target_table = state.trace.tables[node.target_class]
        if !haskey(state, idx)
            # No foreign key has yet been proposoed -- sample from prior or use retained.
            if isnothing(retained_trace)
                # TODO: speed this up if necessary
                # (Will be slow for blocks that contain unconstrained reference slots.)
                existing_lprobs, new_lprob = pitman_yor_prior_logprobs(target_table)
                weights = exp.([values(existing_lprobs)..., new_lprob])
                sampled_index = rand(Categorical(weights))
                state[idx] = sampled_index <= length(existing_lprobs) ? collect(keys(existing_lprobs))[sampled_index] : pclean_gensym!("row")
            else
                state[idx] = retained_trace[idx]
            end
        else
            # There is a foreign key; increment p
            foreign_key = state[idx]
            log_denominator = log(target_table.total_references[] + target_table.pitman_yor_params.strength)
            if haskey(target_table.rows, foreign_key)
                prob = log(target_table.reference_counts[foreign_key] - target_table.pitman_yor_params.discount) - log_denominator
                p += prob
            else
                prob = log(target_table.pitman_yor_params.discount * length(target_table.rows) + target_table.pitman_yor_params.strength) - log_denominator
                p += prob
            end
        end
    end
    
    function process_node!(node::SubmodelNode, idx)
        foreign_key_node_id = node.foreign_key_node_id
        foreign_key = state[foreign_key_node_id]
        target_table = state.trace.tables[strip_subnodes(class_model.nodes[foreign_key_node_id]).target_class]
        if !haskey(target_table.rows, foreign_key)
            process_node!(node.subnode, idx)
        elseif !haskey(state, idx)
            chosen_row = target_table.rows[foreign_key]
            chosen_value = chosen_row[node.subnode_id]
            state[idx] = chosen_value
        end
    end
    
    # Propose values for all non-super-model-nodes.
    i = 1
    while i <= length(vertex_order) && !(class_model.nodes[vertex_order[i]] isa SupermodelNode)
        process_node!(class_model.nodes[vertex_order[i]], vertex_order[i])
        i += 1
    end
    
    # Accumulate weights from supermodel nodes.
    path = nothing
    while i <= length(vertex_order)
        v = vertex_order[i]
        n = class_model.nodes[v]
        
        path = n.path
        source_class = path[end].class
        source_trace = state.trace.tables[source_class]
        vmap = class_model.incoming_references[path]

        next_i = i + 1
        for referring_row_key in state.referring_rows[path]
            state.active_parent_trace = nothing
            state.parent_trace_recomputed = Dict()

            # Initialize parent_trace_recomputed based on values computed at this level.
            for (k, l) in vmap
                state.parent_trace_recomputed[l] = haskey(state, k) ? state[k] : nothing
            end

            state.active_parent_trace = source_trace.rows[referring_row_key]
            j = i
            while j <= length(vertex_order) && class_model.nodes[vertex_order[j]].path == path
                node = class_model.nodes[vertex_order[j]]
                process_node!(node.supernode, node.supernode_id)
                j += 1
            end
            # TODO: It seems bad to set this anew each iteration through the loop...
            # desired behavior is to remember where to resume once all rows have been processed.
            next_i = j
        end
        i = next_i

        state.active_parent_trace = nothing
        state.parent_trace_recomputed = Dict()
    end
    
    return p, q_cont
end

# Modifies `state`
function make_block_proposal!(state::ProposalRowState, block_index::Int, config::InferenceConfig)
    class_model = state.trace.model.classes[state.class]
    block_vertices = class_model.blocks[block_index]
    block_plan = class_model.plans[block_index]
    
    q_disc = 0.0

    # Data-driven proposal
    if config.use_dd_proposals
        observation_indices = keys(state.row_trace)
        compiled_proposal = get!(class_model.compiled_proposals[block_index], observation_indices) do 
            # Pruning may be necessary even though we later compile.
            block_plan = prune_plan(block_plan, state)
            compile_proposal(state.trace.model, state.class, block_plan, observation_indices)
        end
        _, t, q_disc = Base.invokelatest(compiled_proposal, state)
        
        if !isnothing(t)
            merge!(state.row_trace, t)
        end
    end
    
    # Fill in anything not handled by the data-driven proposal:
    #   * everything, if config.use_dd_proposals is false; 
    #   * continuous choices or infinite discrete choices with no preferred values
    #   * anything with no informative observations (and thus pruned from the enumeration plan). 
    # Evaluates `q` for all such choices, and also `p` for the entire proposal.
    p, q_cont = propose_non_enumerable!(block_vertices, state)
    
    #println("Components: p=$p, q_disc=$q_disc, q_cont=$q_cont")
    return state, p - q_disc - q_cont
end
