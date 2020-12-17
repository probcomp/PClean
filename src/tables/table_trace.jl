"""
    Key

A Key identifies a particular record in a table: it is
either an integer (in the context of an observed table) or a symbol
(for a learned table).
"""
const Key = Union{Int, Symbol}

"""
    RowTrace

A RowTrace represents a record of the latent database.
It maps VertexIDs in a PCleanClass to values.
"""
const RowTrace = Dict{VertexID, Any}

"""
    TableTrace(model, parameters, observations, rows, parent, parent_row_ids, reference_counts)

A TableTrace represents the collection of currently instantiated records
for a particular PClean class, i.e. the Pitman-Yor atoms.
"""
struct TableTrace
    # Pitman-Yor parameter values
    pitman_yor_params :: PitmanYorParams
  
    # Value of all parameter nodes (global across rows)
    parameters        :: Dict{VertexID, Parameter}

    # Actual traces of each row, and partial traces containing observations
    rows               :: Dict{Key, RowTrace}
    observations       :: Dict{Key, RowTrace}
    # observation_counts tracks the number of reference slots that directly refer to the row with key k 
    # and imply an observation of the value of its vertex v. Note that a single referring record may refer twice to 
    # the same target record; this counts as two observations.
    observation_counts :: Dict{Key, Dict{VertexID, Int}}
    hashed_keys        :: Dict{Any, Set{Key}}
  
    # Tracking references.
    direct_incoming_references :: Dict{Key, Dict{AbsoluteVertexID, Set{Key}}}   # maps each row here to a dictionary mapping reference slots *targeting* this class to a set of *source* class record keys.
    reference_counts :: Dict{Key, Int}  # total count of foreign keys with this row as their direct target
    total_references :: Ref{Int}        # sum of all reference counts. We maintain this separately to speed up Chinese Restaurant Process probability calculations.
end


struct PCleanTrace
    model  :: PCleanModel
    tables :: Dict{ClassID, TableTrace}
end

# TODO: This is slower than necessary.
function pitman_yor_prior_logprobs(table::TableTrace)
    prior = table.pitman_yor_params
    total_count = table.total_references[]
    logdenom = log(total_count + prior.strength)
    probs = Dict{Symbol, Float64}(i => log(count - prior.discount) - logdenom
                                  for (i, count) in table.reference_counts)
    new_table_prob = log(length(probs) * prior.discount + prior.strength) - logdenom
    return probs, new_table_prob
end
