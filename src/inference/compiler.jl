
# We can try to implement the generated function machinery manually.
# Do a preprocessing pass where we collect the different observation index sets,
# and dispatch manually based on that.
# Use `eval` to generate the code.
# Note: how will this work for latent object proposals?
# Maybe we will need to maintain for every row an observation set, and
# actually look up the appropriate proposals.

# Five cases to handle:

# Maybe we should actually just work with a running score
# and trace and q, and update these "on the way down".
# One issue is that at branch and combine points, it's inaccurate
#  to assume that 0 is where it can be reset to. 
# So in those situations, it may make sense to create dummy
# score and trace and q variables, and after each run, merge them
# into the ones being created. Yes, this seems right...

# A JuliaNode step simply assigns a variable.
# A RandomChoiceNode
#    * Observed: Add log likelihood to score, process the rest, and subtract from score.
#                Maintain the same "immediate next step."
#    * Unobserved: 
#                Add a line for determining the options and prior probabilities.
#                Add a buffer for storing the probabilities of each branch,
#                the probabilities of this trace, 
#                For each option, process the rest of the plan with rests_of_trace
# A SubmodelNode
# A ForeignKeyNode
# # A SupermodelNode

# We define these in new_enumeration currently, so don't need them here (yet)
strip_subnodes(node::SubmodelNode) = strip_subnodes(node.subnode)
strip_subnodes(node) = node

# For now, try to produce code from a model, a block plan, and a set of observed indices (namedtuple?)
function generate_code(model::PCleanModel, class::ClassID, plan::Plan, observation_indices)

    initialization_statements = Expr[]
    # We will generate code here that checks if there is a retained particle, and if there is,
    # sets all these to their values inside the retained particle; otherwise set them all to `nothing`.
    retained_variable_names = Dict{Int, Symbol}()
    variable_names = Dict{Union{Int, Symbol}, Symbol}()

    active_child_traces = Dict{Int, Symbol}()
    active_parent_trace_var = gensym("active_parent_trace")
    in_supernode_loop = false
    recomputed_supernode_variables = Dict{Int, Symbol}()

    # Inputs to the function we will create
    state_var = gensym("state")

    function any_unavailable(arg_nums)
        return any(k -> !(k in observation_indices) && !haskey(variable_names, k), arg_nums)
    end

    function get_arg_names!(arg_nums)
        if any_unavailable(arg_nums)
            return nothing
        end

        for k in arg_nums
            if !haskey(variable_names, k)
                variable_names[k] = gensym()
                push!(initialization_statements, :($(variable_names[k]) = $(state_var)[$k]))
            end
        end

        return [variable_names[k] for k in arg_nums]
    end

    function process_node!(node::JuliaNode, idx, body_statements, prob_output_var, trace_output_var, q_output_var, plan)
        # If it's impossible to compute this node's value, just move on.
        arg_names = get_arg_names!(node.arg_node_ids)
        if isnothing(arg_names)
            return process_plan!(plan, body_statements, prob_output_var, trace_output_var, q_output_var)
        end

        # Otherwise, call the Julia function in a statement.
        new_var_name = get!(gensym, variable_names, idx)
        push!(body_statements, :($new_var_name = $(node.f)($(arg_names...))))
        process_plan!(plan, body_statements, prob_output_var, trace_output_var, q_output_var)
        delete!(variable_names, idx)
    end


    function process_node!(node::RandomChoiceNode, idx, body_statements, prob_output_var, trace_output_var, q_output_var, plan)
        # If this node is not observed and there's no proposal, skip and move on
        if !(idx in observation_indices) && !has_discrete_proposal(node.dist)
            return process_plan!(plan, body_statements, prob_output_var, trace_output_var, q_output_var)
        end
        
        # If it's impossible to compute this node's value, just move on
        arg_names = get_arg_names!(node.arg_node_ids)
        if isnothing(arg_names)
            return process_plan!(plan, body_statements, prob_output_var, trace_output_var, q_output_var)
        end

        # A name for this observation
        new_var_name = get!(gensym, variable_names, idx)

        # If the node is observed, we want to process the rest of the plan, then
        # add our increment to the score.
        if idx in observation_indices
            push!(initialization_statements, :($new_var_name = $(state_var)[$idx]))
            process_plan!(plan, body_statements, prob_output_var, trace_output_var, q_output_var)
            push!(body_statements, :($prob_output_var += PClean.logdensity($(node.dist), $new_var_name, $(arg_names...))))
            return
        end

        # If the node is unobserved we have a more complicated job.
        # First, we create some storage to hold the partial traces and scores associated
        # with each possible future.
        prob_list_var = gensym("prob_list")
        trace_list_var = gensym("trace_list")
        q_list_var = gensym("q_list")
        push!(initialization_statements, :($prob_list_var = Float64[]))
        push!(initialization_statements, :($trace_list_var = Dict{Int, Any}[]))
        push!(initialization_statements, :($q_list_var = Float64[]))
        # Add code to look up the options to iterate through
        options_var, prior_lprobs_var = gensym(), gensym()
        push!(body_statements, :(($options_var, $prior_lprobs_var) = PClean.discrete_proposal($(node.dist), $(arg_names...))))
        # Set up retained value stuff
        retained_value_var = get!(gensym, retained_variable_names, idx)
        chosen_index_var = gensym("chosen_index")
        push!(body_statements, :($chosen_index_var = 0::Int))
        iter_var = gensym("iter")
        # Add a for loop: for (iter_var, new_var_name) in enumerate(options_var)...end
        for_loop_body = Expr[]
        # Handle dummy values:
        push!(for_loop_body, :(if $new_var_name isa PClean.ProposalDummyValue; $new_var_name = PClean.discrete_proposal_dummy_value($(node.dist), $(arg_names...)); end))
        # Handle retained value:
        push!(for_loop_body, :(if $new_var_name == $retained_value_var; $chosen_index_var = $iter_var; end))
        # Process rest of plan
        process_plan!(plan, for_loop_body, prob_output_var, trace_output_var, q_output_var)
        # Add to lists
        push!(for_loop_body, :(push!($prob_list_var, $prob_output_var + $prior_lprobs_var[$iter_var])))
        push!(for_loop_body, :(push!($trace_list_var, $trace_output_var)))
        push!(for_loop_body, :(push!($q_list_var, $q_output_var)))
        push!(for_loop_body, :($prob_output_var = 0.0))
        push!(for_loop_body, :($q_output_var = 0.0))
        # TODO: This isempty check is not necessary to do dynamically; it should be known,
        # based on whether there are any ForeignKeyNode or RandomChoiceNodes yet to come.
        push!(for_loop_body, :(if !isempty($trace_output_var); $trace_output_var = Dict{Int, Any}(); end))
        push!(body_statements, :(for ($iter_var, $new_var_name) in enumerate($options_var); $(for_loop_body...); end))
        # Now, make a choice.
        push!(body_statements, quote 
            $prob_output_var = PClean.logsumexp($prob_list_var)
            $prob_list_var .-= $prob_output_var
            if $chosen_index_var == 0
                $chosen_index_var = Distributions.rand(Distributions.Categorical(exp.($prob_list_var)))
            end
            $trace_output_var = $trace_list_var[$chosen_index_var]
            $trace_output_var[$idx] = $options_var[$chosen_index_var]
            $q_output_var = $q_list_var[$chosen_index_var] + $prob_list_var[$chosen_index_var]
            empty!($prob_list_var)
            empty!($trace_list_var)
            empty!($q_list_var)
        end)
        delete!(variable_names, idx)
    end

    function process_node!(node::ForeignKeyNode, idx, body_statements, prob_output_var, trace_output_var, q_output_var, plan)
        # Make sure we have a reference to the underlying table
        learned_table_var = gensym("learned_table")
        variable_names[node.target_class] = learned_table_var
        push!(initialization_statements, :($learned_table_var = $state_var.trace.tables[$(Meta.quot(node.target_class))]))

        # Determine valid keys for enumeration
        # We actually know at compile-time whether it will be possible to hash,
        # and we can precompute the hash key as part of initialization. (We could even pre-lookup
        # the set. Though, this doesn't handle a dynamically changing learned table during
        # sampling.)
        submodel_hash_keys = model.classes[node.target_class].hash_keys
        can_check_hashed_collection = !isempty(submodel_hash_keys) && all(x -> node.vmap[x] in observation_indices, submodel_hash_keys)
        keys_to_check_var = gensym("keys_to_check")
        if can_check_hashed_collection
            hash_key_var = gensym("hash_key")
            push!(initialization_statements, :($hash_key_var = [   $(  [:($state_var[$(node.vmap[i])]) for i in submodel_hash_keys]...  ) ]))
            push!(initialization_statements, :($keys_to_check_var = collect(get($learned_table_var.hashed_keys, $hash_key_var, Set()))))
        else
            push!(initialization_statements, :($keys_to_check_var = collect(keys($learned_table_var.rows))))
        end
        num_to_check_var = gensym("num_to_check")
        push!(initialization_statements, :($num_to_check_var = length($keys_to_check_var)))
        #push!(initialization_statements, :(print("Num to check is: "); println($num_to_check_var)))
        # We can materialize an array of prior probabilities (Pitman-Yor)
        # once, up front, including for the extra symbol.
        # We can also pre-generate the foreign_row_key / check if it's been retained.
        # (This involves checking that learned table rows has retained value as key or not; if yes, we gensym).
        # then we have something much like the random choice nodes.
        # I think it will actually make sense to do this in two stages: the "cases 1, 2, 3" 
        # should be statically knowable. First we do a loop over the non-new entries, then we can
        # do a marginalization loop to determine "new row" prob. This will duplicate code in the expanded
        # version but I don't see a way around that.

        py_prior_logprobs_var = gensym("py_prior_logprobs")
        push!(initialization_statements, quote
            total_referents = $learned_table_var.total_references[]
            log_denominator = log(total_referents + $learned_table_var.pitman_yor_params.strength)
            $py_prior_logprobs_var = [log($learned_table_var.reference_counts[k] - $learned_table_var.pitman_yor_params.discount) - log_denominator for k in $keys_to_check_var]
            push!($py_prior_logprobs_var, log($learned_table_var.pitman_yor_params.strength + $learned_table_var.pitman_yor_params.discount * length($learned_table_var.rows)) - log_denominator)
        end)

        # Set up accumulation variables.
        prob_list_var = gensym("prob_list")
        trace_list_var = gensym("trace_list")
        q_list_var = gensym("q_list")
        push!(initialization_statements, :($prob_list_var = Float64[]))
        push!(initialization_statements, :($trace_list_var = Dict{Int, Any}[]))
        push!(initialization_statements, :($q_list_var = Float64[]))

        # Set up retained value stuff.
        # Namely: the "new row" ID will either be gensym'd, or, if this is the retained particle,
        # and the target table does not yet have the key that is referenced by the retained particle,
        # it will be that ID.
        retained_value_var = get!(gensym, retained_variable_names, idx)
        push!(initialization_statements, quote
            if isnothing($retained_value_var) || haskey($learned_table_var.rows, $retained_value_var)
                push!($keys_to_check_var, PClean.pclean_gensym!("row"))
            else
                push!($keys_to_check_var, $retained_value_var)
            end
        end)

        chosen_index_var = gensym("chosen_index")
        push!(body_statements, :($chosen_index_var = 0::Int))
        iter_var = gensym("iter")

        # Add a for loop: for iter_var in 1:num_to_check...end
        for_loop_body = Expr[]
        # Set foreign key var
        fk_var_name = get!(gensym, variable_names, idx)
        push!(for_loop_body, :($fk_var_name = $keys_to_check_var[$iter_var]))
        # Handle retained value -- force this choice if need be:
        push!(for_loop_body, :(if $fk_var_name == $retained_value_var; $chosen_index_var = $iter_var; end))
        # Process rest of plan, to see consequences of this chocie
        child_trace_var = get!(gensym, active_child_traces, idx)
        push!(for_loop_body, :($child_trace_var = $learned_table_var.rows[$fk_var_name]))
        process_plan!(plan, for_loop_body, prob_output_var, trace_output_var, q_output_var)
        delete!(active_child_traces, idx)
        # # Add to lists
        push!(for_loop_body, :(push!($prob_list_var, $prob_output_var + $py_prior_logprobs_var[$iter_var])))
        push!(for_loop_body, :(push!($trace_list_var, $trace_output_var)))
        push!(for_loop_body, :(push!($q_list_var, $q_output_var)))
        push!(for_loop_body, :($prob_output_var = 0.0))
        push!(for_loop_body, :($q_output_var = 0.0))
        # TODO: This isempty check is not necessary to do dynamically; it should be known,
        # based on whether there are any ForeignKeyNode or RandomChoiceNodes yet to come.
        push!(for_loop_body, :(if !isempty($trace_output_var); $trace_output_var = Dict{Int, Any}(); end))
        push!(body_statements, :(for $iter_var in 1:$num_to_check_var; $(for_loop_body...); end))

        # Before sampling, do blind generation.
        # Even if earlier, there were no random choices in this trace, there might be when 
        # generating blind, so use new memory.
        push!(body_statements, :($trace_output_var = Dict{Int, Any}()))
        push!(body_statements, :($fk_var_name = $keys_to_check_var[end]))
        push!(body_statements, :(if $fk_var_name == $retained_value_var; $chosen_index_var = $num_to_check_var + 1; end))
        process_plan!(plan, body_statements, prob_output_var, trace_output_var, q_output_var)
        push!(body_statements, :(push!($prob_list_var, $prob_output_var + $py_prior_logprobs_var[end])))
        push!(body_statements, :(push!($trace_list_var, $trace_output_var)))
        push!(body_statements, :(push!($q_list_var, $q_output_var)))
        
        # Now do sampling.
        push!(body_statements, quote
            $prob_output_var = PClean.logsumexp($prob_list_var)
            $prob_list_var .-= $prob_output_var
            if $chosen_index_var == 0
                $chosen_index_var = Distributions.rand(Distributions.Categorical(exp.($prob_list_var)))
            end
            $trace_output_var = $trace_list_var[$chosen_index_var]
            $trace_output_var[$idx] = $keys_to_check_var[$chosen_index_var]
            $q_output_var = $q_list_var[$chosen_index_var] + $prob_list_var[$chosen_index_var]
            empty!($prob_list_var)
            empty!($trace_list_var)
            empty!($q_list_var)
        end)
        delete!(variable_names, idx)
    end

    can_process(node::JuliaNode, idx) = !any_unavailable(node.arg_node_ids)
    can_process(node::RandomChoiceNode, idx) = !any_unavailable(node.arg_node_ids) && (idx in observation_indices || has_discrete_proposal(node.dist))
    can_process(node::ForeignKeyNode, idx) = true # In HDP version, this could be made false by not knowing the argument to the class
    can_process(node::SubmodelNode, idx) = idx in observation_indices || can_process(node.subnode, idx)

    function process_node!(node::SubmodelNode, idx, body_statements, prob_output_var, trace_output_var, q_output_var, plan)
        # We should be able to tell statically which case we're in.
        subnode = node.subnode
        fk_idx = node.foreign_key_node_id

        if !can_process(node, idx)
            return process_plan!(plan, body_statements, prob_output_var, trace_output_var, q_output_var)
        end


        already_exists = haskey(active_child_traces, fk_idx)
        
        # Case 1: We are generating a new child trace.
        if !already_exists
            return process_node!(subnode, idx, body_statements, prob_output_var, trace_output_var, q_output_var, plan)
        end


        fk_var = variable_names[fk_idx]
        learned_table_var = variable_names[strip_subnodes(model.classes[class].nodes[fk_idx]).target_class]
        
        new_var_name = get!(gensym, variable_names, idx)

        # Case 2: this node is observed
        if idx in observation_indices
            push!(initialization_statements, :($new_var_name = $state_var[$idx]))
            rest_of_plan = Expr[]
            process_plan!(plan, rest_of_plan, prob_output_var, trace_output_var, q_output_var)
            push!(body_statements, quote
                chosen_value = $(active_child_traces[fk_idx])[$(node.subnode_id)]
                close_enough = ismissing($new_var_name) && ismissing(chosen_value) || (chosen_value isa Real && isapprox(chosen_value, $new_var_name)) || (!ismissing(chosen_value) && !ismissing($new_var_name) && chosen_value == $new_var_name)
                if !close_enough
                    $prob_output_var = -Inf
                    $q_output_var = -Inf
                else
                    $(rest_of_plan...)
                end
            end)
            return
        end

        # Case 3: node is not observed. We want to copy from the value in the child trace.
        # Emit code to set the variable and continue
        push!(body_statements, :($new_var_name = $(active_child_traces[fk_idx])[$(node.subnode_id)]))
        process_plan!(plan, body_statements, prob_output_var, trace_output_var, q_output_var)
        delete!(variable_names, idx)
    end

    function get_arg_names_supernode(arg_nums)
        return [get(recomputed_supernode_variables, i, :($active_parent_trace_var[$i])) for i in arg_nums]
    end

    function process_node!(node::SupermodelNode, idx, body_statements, prob_output_var, trace_output_var, q_output_var, plan)
        # Some reimagining might be necessary here.
        # As with submodel, we can decide up front which traces we need to enumerate through at each level.
        # We may need to rethink how things work a bit.
        if in_supernode_loop
            if node.supernode isa JuliaNode
                var_name = get!(gensym, recomputed_supernode_variables, node.supernode_id)
                arg_exprs = get_arg_names_supernode(node.supernode.arg_node_ids)
                push!(body_statements, :($var_name = $(node.supernode.f)($(arg_exprs...))))
                return process_plan!(plan, body_statements, prob_output_var, trace_output_var, q_output_var)
            elseif node.supernode isa RandomChoiceNode
                arg_exprs = get_arg_names_supernode(node.supernode.arg_node_ids)
                process_plan!(plan, body_statements, prob_output_var, trace_output_var, q_output_var)
                push!(body_statements, :($prob_output_var += PClean.logdensity($(node.supernode.dist), $(active_parent_trace_var)[$(node.supernode_id)], $(arg_exprs...))))
                return
            elseif node.supernode isa ForeignKeyNode
                @assert false "There should not be a SupermodelNode{ForeignKeyNode}."
            end
        end
        
        # Otherwise, loop through parent traces.
        # It happens to be the case that this loop will only include blocks
        # at a particular level, due to the way that plans are constructed;
        # this may not be an ideal thing to rely on in the future, though.
        # (Briefly: supernodes cannot point to other levels' supernodes in the graph,
        # and supernodes all come at the end. So the levels form connected components
        # separately after the main nodes have all been sampled.)
        for_loop_body = Expr[]
        in_supernode_loop = true
        parent_row_id_var   = gensym("parent_row_id_var")
        source_class = node.path[end].class
        push!(for_loop_body, :($active_parent_trace_var = $state_var.trace.tables[$(Meta.quot(source_class))].rows[$parent_row_id_var]))
        # Make available any relevant changes from the regular nodes.
        vmap = model.classes[class].incoming_references[node.path]
        assignments = Expr[:($(get!(gensym, recomputed_supernode_variables, j)) = $(variable_names[i])) for (i, j) in vmap if haskey(variable_names, i)]
        push!(for_loop_body, quote
            $(assignments...)
        end)
        
        # Call recursively on self, but behavior will be different because we are in_supernode_loop.
        process_node!(node, idx, for_loop_body, prob_output_var, trace_output_var, q_output_var, plan)
        push!(body_statements, :(for $parent_row_id_var in $state_var.referring_rows[$(node.path)]; $(for_loop_body...); end))
        in_supernode_loop = false
        empty!(recomputed_supernode_variables)
    end


    function process_step!(step, body_statements, prob_output_var, trace_output_var, q_output_var)
        process_node!(model.classes[class].nodes[step.idx], step.idx, body_statements, prob_output_var, trace_output_var, q_output_var, step.rest)
    end

    # The function itself will get access to a state that contains
    # information on, e.g., the parent traces of an object.
    # But in what form? Let's first worry about how to handle a simple subset.

    # Note that the compiler *can* be recursive.

    function process_plan!(plan, body_statements, prob_output_var, trace_output_var, q_output_var)
        steps = plan.steps
        if isempty(steps)
            return
        end
        if length(steps) == 1
            return process_step!(steps[1], body_statements, prob_output_var, trace_output_var, q_output_var)
        end

        for step in steps
            # Create new variables to use for each step
            p_var  = gensym("p")
            t_var  = gensym("t")
            tp_var = gensym("tp")
            # Note: it is ok to reuse storage for these, because we are merging into
            # a different dict at the end.
            push!(initialization_statements, :($t_var = Dict{Int, Any}()))
            push!(body_statements, :($p_var = 0.0))
            push!(body_statements, :($(tp_var) = 0.0))
            process_step!(step, body_statements, p_var, t_var, tp_var)
            push!(body_statements, :($prob_output_var += $p_var))
            push!(body_statements, :($q_output_var += $tp_var))
            push!(body_statements, :(merge!($trace_output_var, $t_var)))
            push!(body_statements, :(empty!($t_var)))
        end
    end

    body_statements = Expr[]
    final_prob_var = gensym("prob")
    final_trace_var = gensym("trace")
    final_q_var = gensym("q")
    push!(initialization_statements, :($final_prob_var = 0))
    push!(initialization_statements, :($final_trace_var = Dict{Int, Any}()))
    push!(initialization_statements, :($final_q_var = 0))

    process_plan!(plan, body_statements, final_prob_var, final_trace_var, final_q_var)

    retained_inits = [:($var = $state_var.retained_trace[$idx]) for (idx, var) in retained_variable_names]
    retained_nothings = [:($var = nothing) for (_, var) in retained_variable_names]

    function_name = gensym()
    return quote
        function $function_name($state_var)
            if isnothing($state_var.retained_trace)
                $(retained_nothings...)
            else
                $(retained_inits...)
            end
            $(initialization_statements...)
            $(body_statements...)
            return $final_prob_var, $final_trace_var, $final_q_var
        end

        $function_name
    end
end

function compile_proposal(model::PCleanModel, class::ClassID, plan::Plan, observation_set)
    eval(generate_code(model, class, plan, observation_set))
end

# What storage do we actually need?
# For any *unobserved* random choice nodes
# and any foreign key nodes, we need 
#   Vectors to store probabilities
#   Vectors to store traces
#   Vectors to store trace probabilities.
# We need variables to hold a running score, a running tprob, and a running trace.
# There's always a running score variable, which is separate from the one created by the brancher
# Hmm, it seems at any branch point we may want score to start at 0,
# then add. Can be seen as -- assign variable for observation, but wait until
# after generated code to score.

