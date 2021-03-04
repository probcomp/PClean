# Supports multiple observed tables, but all of them must be 
# "observation classes" with no incoming references.
function initialize_trace(observations::Vector{ObservedDataset}, config::InferenceConfig)
    model = first(observations).query.model
    trace = PCleanTrace(model, Dict())

    # Initialize all classes
    for (class, class_model) in model.classes
        parameters = Dict(i => node.make_parameter() for (i, node) in enumerate(class_model.nodes) if node isa ParameterNode)
        trace.tables[class] = TableTrace(class_model.initial_pitman_yor_params, parameters, Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), Ref(0))
    end

    # Incorporate observations
    for (dataset_num, observed_dataset) in enumerate(observations)
        class = observed_dataset.query.class
        num_rows_total = length(eachrow(observed_dataset.data))
        obs   = trace.tables[class].observations
        sizehint!(obs, num_rows_total)
        rough_n_obs = length(observed_dataset.query.obsmap)
        for (i, row) in enumerate(eachrow(observed_dataset.data))
            row_trace = Dict{VertexID, Any}()
            sizehint!(row_trace, rough_n_obs)
            for (k, v) in zip(propertynames(observed_dataset.data), row)
                if haskey(observed_dataset.query.obsmap, k)
                    node_id = observed_dataset.query.obsmap[k]
                    if !ismissing(v)
                        row_trace[node_id] = v
                    elseif node_id != observed_dataset.query.cleanmap[k] && supports_explicitly_missing_observations(strip_subnodes(model.classes[class].nodes[node_id]).dist)
                        # TODO: use a different missingness marker?
                        row_trace[node_id] = v
                    end
                end
            end
            obs[i] = row_trace

            # Incorporate the row
            run_smc!(trace, class, i, config)

            # Rejuvenate parameters
            if i % config.rejuv_frequency == 0
                for (rejuv_class, rejuv_class_model) in model.classes
                    for (k, param) in trace.tables[rejuv_class].parameters
                        resample_value!(param)
                    end
                    resample_py_params!(trace.tables[rejuv_class])
                end
            end

            # Verbose
            if i % config.reporting_frequency == 0
                println("Initializing row $i of $(num_rows_total) in observations for $class (dataset $dataset_num of $(length(observations)))...")
            end

        end
    end

    return trace
end
  
function pgibbs_sweep!(trace::PCleanTrace, config::InferenceConfig)
    # For each class
    for class in trace.model.class_order
        table = trace.tables[class]
        n_rows = length(table.rows)
        # Update each row
        for (i, key) in enumerate(keys(table.rows))
            if i % config.reporting_frequency == 0
                println("$class: Cleaning row $i of $n_rows")
            end

            # Maybe update parameters
            if i % config.rejuv_frequency == 0
                for (_, param) in table.parameters
                    resample_value!(param)
                end
                resample_py_params!(table)
            end
            run_smc!(trace, class, key, config)
        end
    end
end

function run_inference!(trace::PCleanTrace, config::InferenceConfig)
    for iter = 1:config.num_iters
        println("Iteration $iter/$(config.num_iters)")
        pgibbs_sweep!(trace, config)
    end
end


export run_inference!, initialize_trace
