struct SMC <: Algorithm
  particles :: Int
end

export SMC

function infer_row_trace(alg::SMC, row, dbm::DBModel)
  initialized = false
  argdiff = Tuple([Gen.UnknownChange; [Gen.NoChange for i=1:length(dbm.params)]])
  state = nothing

  for k in dbm.namesInOrder
    if k in keys(row) && !ismissing(row[k])
      addr = dbm.alignment[k]
      obs = choicemap(addr => row[k])
      if !initialized
        state = initialize_particle_filter(dbm.model, Tuple([addr; dbm.params]), obs, alg.particles)
        initialized = true
      else
        particle_filter_step!(state, Tuple([addr; dbm.params]), argdiff, obs)
      end
      maybe_resample!(state)
    end
  end

  # Note: the code below deterministically chooses the best sample, rather than
  # doing true SMC at the  last step.
  # For doing true SMC, use this instead:
  # sampled_trace = sample_unweighted_traces(state, 1)
  sampled_trace = sort(get_traces(state), by=get_score)[end]
  final_trace, = generate(dbm.model, Tuple([nothing; dbm.params]), get_choices(sampled_trace))
  return final_trace
end
