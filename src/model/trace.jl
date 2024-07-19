"""
    Key

A Key identifies a particular record in a table: it is
either an integer (in the context of an observed table) or a symbol
(for a learned table).
"""
const Key = Union{Int,Symbol}

"""
    RowTrace

A RowTrace represents a record of the latent database.
It maps VertexIDs in a PCleanClass to values.
"""
const RowTrace = Dict{VertexID,Any}

"""
    TableTrace(model, parameters, observations, rows, parent, parent_row_ids, reference_counts)

A TableTrace represents the collection of currently instantiated records
for a particular PClean class, i.e. the Pitman-Yor atoms.
"""
struct TableTrace
    # Pitman-Yor parameter values
    pitman_yor_params::PitmanYorParams

    # Value of all parameter nodes (global across rows)
    parameters::Dict{VertexID,Parameter}

    # Actual traces of each row, and partial traces containing observations
    rows::Dict{Key,RowTrace}
    observations::Dict{Key,RowTrace}
    # observation_counts tracks the number of reference slots that directly refer to the row with key k 
    # and imply an observation of the value of its vertex v. Note that a single referring record may refer twice to 
    # the same target record; this counts as two observations.
    observation_counts::Dict{Key,Dict{VertexID,Int}}
    hashed_keys::Dict{Any,Set{Key}}

    # Tracking references.
    direct_incoming_references::Dict{Key,Dict{AbsoluteVertexID,Set{Key}}}   # maps each row here to a dictionary mapping reference slots *targeting* this class to a set of *source* class record keys.
    reference_counts::Dict{Key,Int}  # total count of foreign keys with this row as their direct target
    total_references::Ref{Int}        # sum of all reference counts. We maintain this separately to speed up Chinese Restaurant Process probability calculations.
end


struct PCleanTrace
    model::PCleanModel
    tables::Dict{ClassID,TableTrace}
end

# TODO: This is slower than necessary.
function pitman_yor_prior_logprobs(table::TableTrace)
    prior = table.pitman_yor_params
    total_count = table.total_references[]
    logdenom = log(total_count + prior.strength)
    probs = Dict{Symbol,Float64}(
        i => log(count - prior.discount) - logdenom for
        (i, count) in table.reference_counts
    )
    new_table_prob = log(length(probs) * prior.discount + prior.strength) - logdenom
    return probs, new_table_prob
end

using Distributions: Gamma, logpdf

function pitman_yor_score(params::PitmanYorParams, reference_counts::Vector{Int})
    logprob = 0.0
    n_references = 0
    for (n_objects, size) in enumerate(reference_counts)
        # The choice to start a new cluster
        logprob +=
            log(n_objects * params.discount + params.strength) -
            log(n_references + params.strength)
        # The choices to join that cluster
        if size > 1
            logprob += sum(
                log(i - params.discount) - log(n_references + i + params.strength) for
                i = 1:size-1
            )
        end
        n_references += size
    end
    return logprob
end

function resample_py_params!(trace::TableTrace)
    # Likelihood can be computed via reference counts.
    counts = collect(values(trace.reference_counts))
    current_params = trace.pitman_yor_params
    old_score = pitman_yor_score(current_params, counts)
    # Propose an update to strength.
    proposed_strength = rand(Gamma(1, 1))
    proposed_params = PitmanYorParams(proposed_strength, current_params.discount)
    new_score = pitman_yor_score(proposed_params, counts)
    # Accept or reject it
    old_q = logpdf(Gamma(1, 1), current_params.strength)
    new_q = logpdf(Gamma(1, 1), proposed_strength)
    alpha = new_score + old_q - old_score - new_q
    if log(rand()) < alpha
        current_params = proposed_params
        old_score = new_score
    end
    # Propose an update to discount
    proposed_discount = rand()
    proposed_params = PitmanYorParams(current_params.strength, proposed_discount)
    new_score = pitman_yor_score(proposed_params, counts)
    # Accept or reject
    alpha = new_score - old_score
    if log(rand()) < alpha
        current_params = proposed_params
    end
    trace.pitman_yor_params.discount = current_params.discount
    trace.pitman_yor_params.strength = current_params.strength
end
