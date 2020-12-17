# Instrumented inference routines that record / log details of the
# inference process, in order to export as JSON for visualization.

# As a first attempt, let's try to record "increments."
# What does an increment look like?
# Perhaps it's a set of new objects to be added.
# I am realizing we may want to show "zooomed in" versions of just the
# relevant rows. All of this can be played around with once I have a basic
# JSON representation and some machinery for animations.

# An increment can be represented as a 'complete row' of inferred values,
# and the dependency logic can be re-implemented in the browser client.
# This may be the simplest thing. 
# This makes it sound like we can just look at a single table trace 
# (the top level) and tell how to animate the entire SMC process. 
# Let's try that for now. When we do rejuvenation moves, we can record them 
# as accepted rejuvenation particles; that we *will* need to handle.

using JSON


function initialize_observed_table_instrumented!(query::Query, data::CSV.DataFrame, config::InferenceConfig, measurement, with_rejuv=false, existing_subtable_traces = TableTrace[])
    # Extract model
    model = query.model
  
    # Initialize all parameters
    parameters = Dict(i => node.make_parameter() for (i, node) in enumerate(model.nodes) if node isa ParameterNode)
    learned_table_nodes = [(i, node) for (i, node) in enumerate(model.nodes) if node isa LearnedTableNode]
    subtables  = initialize_subtables_from_existing_traces(existing_subtable_traces, learned_table_nodes)
    # if isnothing(existing_subtable_traces)
    #   subtables  = Dict(i => initialize_learned_table(node.submodel)
    #                     for (i, node) in enumerate(model.nodes) if node isa LearnedTableNode)
    # else 
    #   subtables = initialize_subtables_from_existing_traces(existing_subtable_traces, learned_table_nodes)
    # end
  
    # Initialize observations
    observations = Dict()
    sizehint!(observations, length(eachrow(data)))
    rough_n_obs = length(query.obsmap)
    for (i, row) in enumerate(eachrow(data))
      d = Dict{Int, Any}()
      sizehint!(d, rough_n_obs)
      for (k, v) in zip(names(data), row)
        if haskey(query.obsmap, k)
          if !ismissing(v)
            d[query.obsmap[k]] = v
          elseif query.obsmap[k] != query.cleanmap[k] && supports_explicitly_missing_observations(strip_subnodes(query.model.nodes[query.obsmap[k]]).dist)
            # TODO: use a different missingness marker?
            d[query.obsmap[k]] = v
          end
        end
      end
      observations[i] = d
    end
    #
    # observations = Dict(i => Dict(model.obsmap[k] => v
    #                               for (k, v) in zip(names(data), row)
    #                               if !ismissing(v) && haskey(model.obsmap, k))
    #                     for (i, row) in enumerate(eachrow(data)))
  
    # TODO: are observation counts needed (trivial "all 1" obs counts?)
    table = TableTrace(model, model.initial_pitman_yor_params, parameters, subtables, Dict(), observations, Dict(), Dict(), Dict(), Dict(), Ref(0))
    times = Float64[]
    evaluations = []
    total_time = 0
    println("Starting to initialize rows...")
    for i=1:length(eachrow(data))
      if i % 5 == 0
        println("Initializing row $i...")
        for (j, subtable) in table.subtables
          for (k, param) in subtable.parameters
            resample_value!(param)
          end
        end
      end
      if i % 100 == 0 && with_rejuv
        pgibbs_sweep_instrumented!(table, TableTrace[], InferenceConfig(1, 2; use_mh_instead_of_pg=true, use_dd_proposals = config.use_dd_proposals), measurement, times, evaluations)
        total_time = times[end]
      end
      total_time += @elapsed run_smc!(table, i, TableTrace[], config)
      push!(times, total_time)
      push!(evaluations, measurement(table, i))
    end
  
    return table, times, evaluations
end


function pgibbs_sweep_instrumented!(table::TableTrace, parent_tables::Vector{TableTrace}, config::InferenceConfig, measurement, times = [], evaluations = [])

  # Update parameters
  for (i, param) in table.parameters
    resample_value!(param)
  end

  # Update subtables
  if config.use_lo_sweeps
    for (i, subtable) in table.subtables
      pgibbs_sweep_instrumented!(subtable, [table, parent_tables...], config, (t, i) -> measurement(table, length(table.rows)), times, evaluations)
    end
  end

  # Update rows
  total_time = isempty(times) ? 0.0 : last(times)
  n_rows = length(table.rows)
  prefix = string(["\t" for _ in parent_tables]...)
  for (idx, key) in enumerate(keys(table.rows))
    if idx % 500 == 0
      println("\tCleaning row $idx/$n_rows...")
      for (i, param) in table.parameters
        resample_value!(param)
      end
    end
    total_time += @elapsed run_smc!(table, key, parent_tables, config)
    push!(times, total_time)
    push!(evaluations, measurement(table, length(table.rows)))
  end

  return times, evaluations
end






evaluations_json(accs) = begin
    accs
end

# We should include a node if it is a RandomChoiceNode, or if it
# is required by some query, or if it is a ForeignKeyNode.
# This is true for SubmodelNodes as well.
# We do not need fine-grained argument information, I think.

# For each table, we need a schema. This includes names for nodes,
# and metadata:
#    ForeignKeyNodes -- what subtable index they refer to
#    SubmodelNodes   -- what foreign key node they refer to, and what index within the submodel
#    Other nodes     -- none?

table_node_json(node::JuliaNode) = Dict(:type => "Attribute")
table_node_json(node::RandomChoiceNode) = Dict(:type => "Attribute")
table_node_json(node::ForeignKeyNode) = Dict(:type => "ForeignKey", :subtable => node.learned_table_node_id) 
table_node_json(node::SubmodelNode) = Dict(:type => "Submodel", :foreign_key_node => node.foreign_key_node_id, :subnode_id => node.subnode_id)


row_json(row, nodes_to_include) = begin

    # Include values for 
    #  * foreign key nodes
    #  * random choice nodes
    #  * selected Julia nodes
    #  * submodel nodes that are selected.
    Dict(k => v for (k, v) in row if k in nodes_to_include)
end


function subtable_query(query, idx)
    subtable_model = query.model.nodes[idx].submodel
    subtable_cleanmap = Dict(k => query.model.nodes[v].subnode_id for (k, v) in query.cleanmap if query.model.nodes[v] isa SubmodelNode && query.model.nodes[query.model.nodes[v].foreign_key_node_id].learned_table_node_id == idx)
    subtable_obsmap = Dict(k => query.model.nodes[v].subnode_id for (k, v) in query.obsmap if query.model.nodes[v] isa SubmodelNode && query.model.nodes[query.model.nodes[v].foreign_key_node_id].learned_table_node_id == idx) 
    Query(subtable_model, subtable_cleanmap, subtable_obsmap)
end

function table_model_json(table_model, query)
    # Which nodes to use depends on the query.
    must_include = union(values(query.cleanmap), values(query.obsmap))
    Dict(
        :names => table_model.names,
        :nodes => [table_node_json(node) for node in table_model.nodes if node isa Union{ForeignKeyNode, RandomChoiceNode} || node in must_include]
    )
end


# Returns JSON for the model and its submodels.
function table_model_json(query)

    subtables = Dict(k => table_model_json(subtable_query(query, k)) for (k, v) in enumerate(query.model.nodes) if v isa LearnedTableNode)

    should_include(id, node::RandomChoiceNode) = true
    should_include(id, node::JuliaNode) = id in values(query.obsmap) || id in values(query.cleanmap)
    should_include(id, node::ForeignKeyNode) = true
    should_include(id, node::SubmodelNode) = begin
        subtable_id = query.model.nodes[node.foreign_key_node_id].learned_table_node_id
        node.subnode_id in keys(subtables[subtable_id][:nodes])
    end
    should_include(id, node) = false

    nodes = Dict(k => table_node_json(v) for (k, v) in enumerate(query.model.nodes) if should_include(k, v))

    Dict(
        :names => Dict(name => k for (name, k) in query.model.names if k in keys(nodes) || k in keys(subtables)),
        :nodes => nodes,
        :subtables => subtables
    )
end


function table_trace_json(table_trace, query)
    
    model_json = table_model_json(query)

    nodes_to_include = keys(model_json[:nodes])

    JSON.json(
        Dict(
            :model => model_json,

            :rows  => Dict(k => row_json(v, nodes_to_include) for (k, v) in table_trace.rows),

            :observations => Dict(k => row_json(v, nodes_to_include) for (k, v) in table_trace.observations)
        )
    )
end

function row_json_in_subtable(table_path, model_json, row)
    if isempty(table_path)
        nodes_to_include = keys(model_json[:nodes])
        return row_json(row, nodes_to_include)
    end

    row_json_in_subtable(table_path[2:end], model_json[:subtables][first(table_path)], row)
end

function rejuvenation_json(table_trace, query, history)
    model_json = table_model_json(query)

    JSON.json([Dict(:key => v[:key], :deletions => v[:deletions], :parent_rows => v[:parent_rows], :table => v[:table], :row => row_json_in_subtable(v[:table], model_json, v[:row]), :obs => row_json_in_subtable(v[:table], model_json, v[:obs])) for v in history])
end


# Problem: I need to generate JSON for rejuvenation moves.
# What should this JSON look like? A first pass is that it should include a class, ID, and new value.
# However, it might also be beneficial to include, directly in the JSON, the classes and IDs of any
# parents, if that's easy to do.


# OK, let's try to pass back other things too.
# Namely, deleted rows. 
function run_smc_instrumented!(table::TableTrace, key::Key, parent_tables::Vector{TableTrace}, config::InferenceConfig, table_path)
    # When `key` does not yet exist in `table`, we are doing
    # vanilla SMC in order to initialize a new row, given observations.
    # Otherwise, we are performing a CSMC update, removing table.rows[key],
    # running SMC, and re-adding.
    is_csmc_run = haskey(table.rows, key)
    retained_row_trace = nothing
    if is_csmc_run
        # Set the retained row trace
        retained_row_trace = table.rows[key]

        # Update all dependency tracking state so that it's as though
        # this row did not exist in `table`.
        deletions = unincorporate_row!(table, key, true, []) # last arg, table_path, is [], because we want relative path.
    end

    # Initialize particles. Even if we are updating
    # an existing row, the particles ignore this fact
    # for now; they all start blank, with only parameters,
    # subtables, and observations filled in.
    starting_values = initialize_row_trace_for_smc(table, key)
    parent_rows = collect_parent_rows(table, key, parent_tables)
    starting_state  = ProposalRowState(table.model, starting_values, key,
                                        parent_tables, parent_rows, nothing)
    particles = [initialize_particle(starting_state) for i=1:config.num_particles]

    # Run the SMC algorithm, one block proposal at a time.
    # TODO: is log_ml meaningful in CSMC run?
    log_ml = 0.0
    for i=1:length(table.model.blocks)
        for j=1:config.num_particles
        if j == 1
            particles[j].state.retained_trace = retained_row_trace
        end
        extend_particle!(particles[j], config)
        end

        # Now perform a "maybe resample" step.
        # TODO: don't resample right before last round?
        particles, log_ml_increment = maybe_resample(particles; retain_first = is_csmc_run)
        log_ml += log_ml_increment
    end

    # At the end, we have a weighted collection of particles; choose 1 to return.
    (log_total_weight, log_normalized_weights) = normalize_weights([particle.weight for particle in particles])
    weights = exp.(log_normalized_weights)
    if config.use_mh_instead_of_pg
        #println(weights)
        #println(particles[2].state.row_trace[20])
        chosen_index = rand(Bernoulli(min(1, weights[2] / (1e-10 + weights[1])))) == 1 ? 2 : 1
    else
        chosen_index = rand(Categorical(weights))
    end
    chosen_row_trace = particles[chosen_index].state.row_trace

    # Any provisional foreign key entries are now added to the table for good
    table.rows[key] = chosen_row_trace
    incorporate_row!(table, key)

    # Update more basic sufficient statistics.
    if is_csmc_run
        if chosen_index != 1
        # Update sufficient statistics, removing old row and adding new one
        update_sufficient_statistics!(table.model, retained_row_trace, :dec)
        update_sufficient_statistics!(table.model, chosen_row_trace,   :inc)
        # const TrackedParent = Vector{Tuple{VertexID, InjectiveVertexMap}}

        # Update all parents.
        # TODO: idea -- instead of storing Set{Key}, store *which* FKnodes
        #       reference the current rows.
        update_parents_with_new_values_for_child_row!(table, key, chosen_row_trace, parent_tables, parent_rows)
        end
    else
        # This is the initial run and there are no parents.
        update_sufficient_statistics!(table.model, chosen_row_trace, :inc)
    end
    return Dict(:table => table_path, :key => key, :parent_rows => deepcopy(parent_rows), :deletions => deletions, :row => chosen_row_trace, :obs => table.observations[key]), log_ml + log_total_weight - log(config.num_particles)
end

function pgibbs_sweep_instrumented_anim!(table::TableTrace, parent_tables::Vector{TableTrace}, config::InferenceConfig, measurement, times = [], evaluations = [], history=[], table_path=[])

    # Update parameters
    for (i, param) in table.parameters
        resample_value!(param)
    end

    # Update subtables
    if config.use_lo_sweeps
        for (i, subtable) in table.subtables
            old_table_path = table_path
            table_path = copy(table_path)
            push!(table_path, i)
            pgibbs_sweep_instrumented_anim!(subtable, [table, parent_tables...], config, (t, i) -> measurement(table, length(table.rows)), times, evaluations, history, table_path)
            table_path = old_table_path
        end
    end

    # Update rows
    total_time = isempty(times) ? 0.0 : last(times)
    n_rows = length(table.rows)
    prefix = string(["\t" for _ in parent_tables]...)
    for (idx, key) in enumerate(keys(table.rows))
        if idx % 500 == 0
            println("\tCleaning row $idx/$n_rows...")
            for (i, param) in table.parameters
                resample_value!(param)
            end
        end
        stats = @timed run_smc_instrumented!(table, key, parent_tables, config, table_path)
        total_time += stats.time

        move_record = stats.value[1]
        push!(history, move_record)
        push!(times, total_time)
        push!(evaluations, measurement(table, length(table.rows)))
    end

    return times, evaluations, history
end