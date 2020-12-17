# State for a row being proposed for a certain class.
mutable struct ProposalRowState
    trace::PCleanTrace
    class::ClassID

    # Mutable -- stores this row trace so far.
    # Is initialized to hold any parameters / observations ---
    # including in submodel nodes.
    row_trace::RowTrace
    row_key::Key
  
    # Which rows refer to this one at each path.
    referring_rows::Dict{Path,Set{Key}}
  
    # Pointer to active parent trace, or nothing if not yet initialized
    active_parent_trace::Union{RowTrace,Nothing}
  
    # When initialized, populate with vmapped contents of row_trace,
    # or `nothing` for any vmap entries for which we don't yet have values.
    # a haskey(...) on this IndividualRowState will return false if we explicitly
    # store nothing here; else return the active_parent_trace's value.
    parent_trace_recomputed::Union{RowTrace,Nothing}
  
    # For particle Gibbs
    retained_trace::Union{RowTrace,Nothing}
  
    function ProposalRowState(model, class, row_trace, row_key, referring_rows, retained_trace=nothing)
        new(model, class, row_trace, row_key, referring_rows, nothing, nothing, retained_trace)
    end
end
  
  
  # Will this be too slow?
function Base.getindex(state::ProposalRowState, i::Int)
    if !isnothing(state.active_parent_trace)
        haskey(state.parent_trace_recomputed, i) && return state.parent_trace_recomputed[i]
        return state.active_parent_trace[i]
    end
    state.row_trace[i]
end
  
function Base.setindex!(state::ProposalRowState, value, i::Int)
    if !isnothing(state.active_parent_trace)
        state.parent_trace_recomputed[i] = value
        return
    end
    state.row_trace[i] = value
end
  
function Base.haskey(state::ProposalRowState, i::Int)
    # No parent trace; row has key
    isnothing(state.active_parent_trace) && haskey(state.row_trace, i) && return true
    # Parent trace and recomputed doesn't cover
    !isnothing(state.active_parent_trace) && !haskey(state.parent_trace_recomputed, i) && haskey(state.active_parent_trace, i) && return true
    # Parent trace and recomputed covers the key
    !isnothing(state.active_parent_trace) && !isnothing(state.parent_trace_recomputed[i]) && return true
    # Otherwise, it's not here
    return false
end
  
function Base.delete!(state::ProposalRowState, i::Int)
    if !isnothing(state.active_parent_trace)
        state.parent_trace_recomputed[i] = nothing
    end
    delete!(state.row_trace, i)
end
  