mutable struct SMCParticle
    state::ProposalRowState
    weight::Float64
    block_index::Int
end

# Initializing a particle
function initialize_particle(state::ProposalRowState)
    SMCParticle(ProposalRowState(state.trace, state.class, 
                                 copy(state.row_trace), state.row_key,
                                 state.referring_rows, nothing),
                                 0.0, 1)
end

# Copying a particle (waste of space?)
# Sets weight to 0
clone_with_zero_weight(p::SMCParticle) = begin
    new_particle = initialize_particle(p.state)
    new_particle.block_index = p.block_index
    return new_particle
end

function collect_referring_rows(trace::PCleanTrace, class::ClassID, key::Key)
    table = trace.tables[class]
    class_model = trace.model.classes[class]

    # This will happen if we are dealing with the observation class
    if !haskey(table.direct_incoming_references, key)
        return Dict{Path,Set{Key}}()
    end

    referring_rows = Dict{Path, Set{Key}}()
    paths = sort!(collect(keys(class_model.incoming_references)), by=length, rev=true)
    while !isempty(paths)
        next_path = pop!(paths)
        last_path, last_link = next_path[1:end-1], next_path[end]
        if isempty(last_path)
            referring_rows[next_path] = table.direct_incoming_references[key][last_link]
        else
            last_referring_rows = referring_rows[last_path]
            last_table = trace.tables[last_path[end].class]
            referring_rows[next_path] = union([last_table.direct_incoming_references[key][last_link] for key in last_referring_rows]...)
        end
    end

    return referring_rows
end

function fill_parameters!(trace::PCleanTrace, class::ClassID, row_trace::RowTrace, vmap_function)
    table = trace.tables[class]
    for (i, param) in table.parameters
        row_trace[vmap_function(i)] = param
    end
    for node in trace.model.classes[class].nodes
        if node isa ForeignKeyNode
            fill_parameters!(trace, node.target_class, row_trace, i -> vmap_function(node.vmap[i]))
        end
    end
end

function initialize_row_trace_for_smc(trace::PCleanTrace, class::ClassID, key::Key)
    table = trace.tables[class]
    row_trace = copy(table.observations[key])
    fill_parameters!(trace, class, row_trace, identity)
    return row_trace
end


function extend_particle!(particle::SMCParticle, config::InferenceConfig)
    new_state, incremental_weight = make_block_proposal!(particle.state, particle.block_index, config)
    particle.weight += incremental_weight
    particle.block_index += 1
    return particle
end

function normalize_weights(log_weights::Vector{Float64})
    log_total_weight = logsumexp(log_weights)
    log_normalized_weights = log_weights .- log_total_weight
    return (log_total_weight, log_normalized_weights)
end

function effective_sample_size(log_normalized_weights::Vector{Float64})
    log_ess = -logsumexp(2. * log_normalized_weights)
    return exp(log_ess)
end

function maybe_resample(particles::Vector{SMCParticle}, ess_threshold::Real=length(particles) / 2; retain_first=false, verbose=false)
    num_particles = length(particles)
    (log_total_weight, log_normalized_weights) = normalize_weights([particle.weight for particle in particles])
    ess = effective_sample_size(log_normalized_weights)
    do_resample = ess < ess_threshold
    if verbose
        println("effective sample size: $ess, doing resample: $do_resample")
    end
    # TODO: This does not use the MH rule -- that is only used at the end. Should using "MH" mean that each block is 
    # separately accepted or rejected (rather than using this stochastic proposal process?)
    if do_resample
        weights = exp.(log_normalized_weights)
        new_indices = retain_first ? [1; rand(Categorical(weights), num_particles - 1)...] : rand(Categorical(weights), num_particles)
        incremental_log_ml = log_total_weight - log(num_particles)
        new_particles = map(clone_with_zero_weight, particles[new_indices])
        return new_particles, incremental_log_ml
    end
    return particles, 0.0
end

# Runs SMC or CSMC.
function run_smc!(trace::PCleanTrace, class::ClassID, key::Key, config::InferenceConfig)
    table = trace.tables[class]

    # When `key` does not yet exist in the table, we are doing
    # vanilla SMC in order to initialize a new row, given observations.
    # Otherwise, we are performing a CSMC update, removing table.rows[key],
    # running SMC, and re-adding.
    is_csmc_run = haskey(table.rows, key)

    # Remove existing row if it exists
    retained_row_trace = nothing
    if is_csmc_run
        # Set the retained row trace
        retained_row_trace = table.rows[key]
        
        # Update all dependency tracking state so that it's as though
        # this row did not exist in `table`.
        unincorporate_row!(trace, class, key)
    end
    
    # Initialize particles. Even if we are updating
    # an existing row, the particles ignore this fact
    # for now; they all start blank, with only parameters
    # and observations filled in.
    starting_values = initialize_row_trace_for_smc(trace, class, key)
    referring_rows = collect_referring_rows(trace, class, key)
    starting_state  = ProposalRowState(trace, class, starting_values, key,
                                        referring_rows, nothing)
    particles = [initialize_particle(starting_state) for i = 1:config.num_particles]
    
    # Run the SMC algorithm, one block proposal at a time.
    log_ml = 0.0
    num_blocks = length(trace.model.classes[class].blocks)
    for i = 1:num_blocks
        for j = 1:config.num_particles
            if j == 1
                particles[j].state.retained_trace = retained_row_trace
            end
            extend_particle!(particles[j], config)
        end
        
        # Don't perform resampling if:
        #  * we are not using PG, or
        #  * this is the last step.
        if !config.use_mh_instead_of_pg && i < num_blocks
            particles, log_ml_increment = maybe_resample(particles; retain_first=is_csmc_run)
            log_ml += log_ml_increment
        end
    end
    
    # At the end, we have a weighted collection of particles; choose 1 to return.
    (log_total_weight, log_normalized_weights) = normalize_weights([particle.weight for particle in particles])
    weights = exp.(log_normalized_weights)
    if config.use_mh_instead_of_pg && is_csmc_run
        chosen_index = rand(Bernoulli(min(1, weights[2] / (1e-10 + weights[1])))) == 1 ? 2 : 1
    else
        chosen_index = rand(Categorical(weights))
    end
    chosen_row_trace = particles[chosen_index].state.row_trace
    
    # Any provisional foreign key entries are now added to the table for good
    table.rows[key] = chosen_row_trace
    incorporate_row!(trace, class, key)
    
    # Update more basic sufficient statistics.
    if is_csmc_run
        if chosen_index != 1
            # Update sufficient statistics, removing old row and adding new one
            update_sufficient_statistics!(trace.model.classes[class], retained_row_trace, :dec)
            update_sufficient_statistics!(trace.model.classes[class], chosen_row_trace,   :inc)
            
            # Update all parents.
            update_referring_rows_with_new_values_for_updated_row!(trace, class, key, chosen_row_trace, referring_rows)
        end
    else
        # This is the initial run and there are no parents.
        update_sufficient_statistics!(trace.model.classes[class], chosen_row_trace, :inc)
    end
    return log_ml + log_total_weight - log(config.num_particles)
end
