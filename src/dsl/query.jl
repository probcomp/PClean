using DataFrames: DataFrame

struct Query
    model::PCleanModel
    class::ClassID
    cleanmap::Dict{Symbol,VertexID}
    obsmap::Dict{Symbol,VertexID}
end

function add_to_query!(query, symbol, clean, dirty)
    clean_id = resolve_dot_expression(query.model, query.class, clean)
    dirty_id = resolve_dot_expression(query.model, query.class, dirty)
    query.cleanmap[symbol] = clean_id
    query.obsmap[symbol] = dirty_id
end

macro query(model, body)
    q = gensym("query")

    statements = map(body.args) do e
        if length(e.args) == 2
            s, a = e.args
            s = s isa String ? Symbol(s) : s
            return :(add_to_query!($q, $(Meta.quot(s)), $(Meta.quot(a)), $(Meta.quot(a))))
        elseif length(e.args) == 3
            s, c, d = e.args
            s = s isa String ? Symbol(s) : s
            return :(add_to_query!($q, $(Meta.quot(s)), $(Meta.quot(c)), $(Meta.quot(d))))
        else
            @error "Syntax error in query: $e"
        end
    end
    quote
        begin
            $q = Query($(esc(model.args[1])), $(model.args[2]), Dict(), Dict())
            $(statements...)
            $q
        end
    end
end

struct ObservedDataset
    query::Query
    data::DataFrame
end

export Query, @query, ObservedDataset
