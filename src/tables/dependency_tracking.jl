# Given a row, either increment or decrement the sufficient statistics
# stored in any parameters used by choices in this row. Note that only
# RandomChoiceNodes can accumulate sufficient statistics; each parameter
# is scoped so that it can only be used at the level where it is introduced,
# so we don't have to worry about statistics coming from supernodes/subnodes.
function update_sufficient_statistics!(model::PCleanClass, row_trace::RowTrace, inc_or_dec::Symbol, reevaluate_jns=false)
    for (i, node) in enumerate(model.nodes)
        if reevaluate_jns && node isa JuliaNode
            evaluated_args = [row_trace[arg_node_id] for arg_node_id in node.arg_node_ids]
            row_trace[i] = node.f(evaluated_args...)
        end
        if node isa RandomChoiceNode
            evaluated_args = [row_trace[arg_node_id] for arg_node_id in node.arg_node_ids]
            if inc_or_dec == :inc
                incorporate_choice!(node.dist, row_trace[i], evaluated_args...)
            elseif inc_or_dec == :dec
                unincorporate_choice!(node.dist, row_trace[i], evaluated_args...)
            end
        end
    end
end


# Update all dependency tracking logic to reflect that this
# row is no longer in the given table.
function unincorporate_row!(trace::PCleanTrace, class::ClassID, row_key::Key, track_deletions=false)
    table     = trace.tables[class]
    model     = trace.model.classes[class]
    row_trace = table.rows[row_key]
    obs       = table.observations[row_key]

    # Update hashes
    if !isempty(model.hash_keys)
        hash_key = [row_trace[i] for i in model.hash_keys]
        @assert haskey(table.hashed_keys, hash_key)

        delete!(table.hashed_keys[hash_key], row_key)
        if isempty(table.hashed_keys[hash_key])
            delete!(table.hashed_keys, hash_key)
        end
    end

    all_deletions = []
    for (i, node) in enumerate(model.nodes)
        if node isa ForeignKeyNode      
            # Collect node IDs, in target table, of fields which are observed in
            # the row being deleted. (Note: if the row is deleted because no 
            # one refers to it, it will not have any observations. But 
            # unincorporate_row! is also called during rejuvenation: a row is 
            # removed, leaving referring tables in an invalid state, and then 
            # reincorporated with its rejuvenated values. During this process,
            # the set of observed values *for* the row being updated stays constant.)
            obs_to_delete = VertexID[target_node_id for (target_node_id, source_node_id) in node.vmap
                                             if haskey(obs, source_node_id)]
            target_row_key = row_trace[i]
            reference_slot_id = AbsoluteVertexID(class, i)
            deletions = unrefer_to_row!(trace, node.target_class, target_row_key, 
                                            reference_slot_id, row_key, 
                                            obs_to_delete, track_deletions)
            if track_deletions
                push!(all_deletions, deletions...)
            end
        end
    end
    return all_deletions
end


# Update all dependency tracking logic to reflect that this row
# has been added to the table.
function incorporate_row!(trace::PCleanTrace, class::ClassID, row_key::Key)
    table     = trace.tables[class]
    model     = trace.model.classes[class]
    row_trace = table.rows[row_key]
    obs       = table.observations[row_key]
  
    # Update hashes
    if !isempty(model.hash_keys)
        hash_key = [row_trace[i] for i in model.hash_keys]
        if !haskey(table.hashed_keys, hash_key) 
            (table.hashed_keys[hash_key] = Set())
        end
        push!(table.hashed_keys[hash_key], row_key)
    end
  
    for (i, node) in enumerate(model.nodes)
        if node isa ForeignKeyNode
            target_row_trace = Dict{VertexID,Any}(target_node_id => row_trace[source_node_id]
                                                    for (target_node_id, source_node_id) in node.vmap)
            target_row_obs   = Dict{VertexID,Any}(target_node_id => obs[source_node_id]
                                                    for (target_node_id, source_node_id) in node.vmap
                                                    if haskey(obs, source_node_id))
            target_row_key   = row_trace[i]
            reference_slot_id = AbsoluteVertexID(class, i)
            refer_to_row!(trace, node.target_class, target_row_key, reference_slot_id,
                                    row_key, target_row_trace, target_row_obs)
        end
    end
end


function unincorporate_observations!(trace::PCleanTrace, class::ClassID, key::Key, observations_to_delete::Vector{VertexID})
    table = trace.tables[class]
    model = trace.model.classes[class]

    # Now that somoe other reference slot no longer refers to me, I need to decrement
    # my observation counts for anything that the referring row observed.
    no_longer_observed_node_ids = Set()
    for observed_node_id in observations_to_delete
        table.observation_counts[key][observed_node_id] -= 1
        if iszero(table.observation_counts[key][observed_node_id])
            push!(no_longer_observed_node_ids, observed_node_id)
            delete!(table.observations[key], observed_node_id)
        end
    end
  
    # For rows that this row refers to, they may also need to decrement 
    # observation counts (for my no_longer_observed_node_ids).
    row_trace = table.rows[key]
    for (i, node) in enumerate(model.nodes)
        if node isa ForeignKeyNode
            target_row_key = row_trace[i]
            observations_to_delete_in_target = [target_node_id
                                                    for (target_node_id, source_node_id) in node.vmap
                                                    if in(source_node_id, no_longer_observed_node_ids)]
            unincorporate_observations!(trace, node.target_class, target_row_key, observations_to_delete_in_target)
        end
    end
end


function incorporate_observations!(trace::PCleanTrace, class::ClassID, key::Key, obs::RowTrace)
    table        = trace.tables[class]
    model        = trace.model.classes[class]
    row_trace    = table.rows[key]
    existing_obs = table.observations[key]

    newly_observed_node_ids = Set()
    for (node_id, observed_value) in obs
        if haskey(existing_obs, node_id)
            table.observation_counts[key][node_id] += 1
        else
            existing_obs[node_id] = observed_value
            push!(newly_observed_node_ids, node_id)
            table.observation_counts[key][node_id] = 1
        end
    end
  
    for (i, node) in enumerate(model.nodes)
        if node isa ForeignKeyNode
            target_row_key = row_trace[i]
            target_obs = Dict{Int,Any}(target_node_id => obs[source_node_id]
                                        for (target_node_id, source_node_id) in node.vmap
                                        if in(source_node_id, newly_observed_node_ids))
            incorporate_observations!(trace, node.target_class, target_row_key, target_obs)
        end
    end
end
  


function unrefer_to_row!(trace::PCleanTrace, target_class::ClassID, target_key::Key, 
                         reference_slot_id::AbsoluteVertexID, referring_key::Key, 
                         observations_to_delete::Vector{VertexID}, track_deletions=false)
    target_table = trace.tables[target_class]
    target_model = trace.model.classes[target_class]
    
    # Remove pointer from target row to referring row
    delete!(target_table.direct_incoming_references[target_key][reference_slot_id], referring_key)
    
    # TODO: Think about whether the following "garbage collection" step is really necessary.
    if isempty(target_table.direct_incoming_references[target_key][reference_slot_id])
        delete!(target_table.direct_incoming_references[target_key], reference_slot_id)
    end

    # If necessary "unobserve" cells of target row
    unincorporate_observations!(trace, target_class, target_key, observations_to_delete)
  
    # Decrement count of total references to this table.
    target_table.total_references[] -= 1

    # If this is not the last referrer, just decrement
    # count and move on.
    if target_table.reference_counts[target_key] > 1
        target_table.reference_counts[target_key] -= 1
        return []
    end

    # Otherwise, this row must be deleted.
    deletions = unincorporate_row!(trace, target_class, target_key, track_deletions)
    update_sufficient_statistics!(target_model, target_table.rows[target_key], :dec)
    # Delete all tracked metadata about this row.
    delete!(target_table.reference_counts, target_key)
    delete!(target_table.rows, target_key)
    delete!(target_table.observations, target_key)
    delete!(target_table.observation_counts, target_key)
    delete!(target_table.direct_incoming_references, target_key)
    if track_deletions
        return [(target_class, target_key), deletions...]
    end
    return []
end


function refer_to_row!(trace::PCleanTrace, target_class::ClassID, 
                        target_key::Key, 
                        reference_slot_id::AbsoluteVertexID, referring_key::Key,
                        row_trace::RowTrace, obs::RowTrace)
    target_table = trace.tables[target_class]
    if !haskey(target_table.rows, target_key)
        # NOTE: assumes row_trace is not shared memory. But it shouldn't be:
        # it is explicitly created (then thrown away) in the caller.
        target_table.rows[target_key] = row_trace
        target_table.reference_counts[target_key] = 0
        target_table.observations[target_key] = Dict()
        target_table.observation_counts[target_key] = Dict()
        target_table.direct_incoming_references[target_key] = Dict(reference_slot_id => Set())
        # Recursively instantiate any additional
        # latent rows in lower tables. But this
        # is *before* the present row has any observations,
        # meaning that lower rows will still need to have
        # any observations incorporated.
        incorporate_row!(trace, target_class, target_key)
        update_sufficient_statistics!(trace.model.classes[target_class], row_trace, :inc)
    end

    target_table.reference_counts[target_key] += 1
    target_table.total_references[] += 1

    # Create new reference set if it doesn't yet exist, and store this reference.
    push!(get!(target_table.direct_incoming_references[target_key],
                reference_slot_id,
                Set()), 
          referring_key)
    incorporate_observations!(trace, target_class, target_key, obs)
end


function update_referring_rows_with_new_values_for_updated_row!(trace::PCleanTrace, 
    class::ClassID, key::Key, new_values::RowTrace,
    referring_rows::Dict{Path, Set{Key}})
    model = trace.model.classes[class]
    for (path, vmap) in model.incoming_references
        referring_class = path[end].class
        referring_table = trace.tables[referring_class]
        relevant_rows  = referring_rows[path]
        for referring_row_key in relevant_rows
            # Unincorporate old values
            update_sufficient_statistics!(trace.model.classes[referring_class], referring_table.rows[referring_row_key], :dec)
            # Update values
            for (target_node_id, source_node_id) in vmap
                referring_table.rows[referring_row_key][source_node_id] = new_values[target_node_id]
            end
            # Reincorporate new values (including recomputing Julia values)
            update_sufficient_statistics!(trace.model.classes[referring_class], referring_table.rows[referring_row_key], :inc, true)
        end
    end
end
