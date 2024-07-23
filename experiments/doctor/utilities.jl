function countmap(samples::Vector{T}) where {T}
    count = Dict{T,Int}()
    for s in samples
        count[s] = get(count, s, 0) + 1
    end
    count
end

function histograms(results)
    physicians = Dict{Symbol,Int}()
    businesses = Dict{Symbol,Int}()
    for r in results
        physician_id = first(r[1])
        if !(physician_id in keys(physicians))
            physicians[physician_id] = 0
        end
        physicians[physician_id] += 1

        business_id = last(r[1])
        if !(business_id in keys(businesses))
            businesses[business_id] = 0
        end
        businesses[business_id] += 1
    end
    physicians, businesses
end

function find_person(trace; firstname = nothing, lastname = nothing)
    firstname === nothing && lastname === nothing && error("Specify at least first or last")
    first_id = PClean.resolve_dot_expression(trace.model, :Physician, :first)
    last_id = PClean.resolve_dot_expression(trace.model, :Physician, :last)
    function f(pair)
        row = last(pair)
        if lastname === nothing
            return row[first_id] == firstname
        elseif firstname === nothing
            return row[last_id] == lastname
        else
            return row[first_id] == firstname && row[last_id] == lastname
        end
    end
    filter(f, trace.tables[:Physician].rows)
end

function find_business(trace, physician_id::Symbol)
    p_id = PClean.resolve_dot_expression(trace.model, :Obs, :p)
    function f(pair)
        row = last(pair)
        return row[p_id] == physician_id
    end
    filter(f, trace.tables[:Obs].rows)
end

function physician_name(data)
    first_id = PClean.resolve_dot_expression(PhysicianModel, :Physician, :first)
    last_id = PClean.resolve_dot_expression(PhysicianModel, :Physician, :last)
    data[first_id], data[last_id]
end


function find_spirit_service(trace)
    rows = trace.tables[:BusinessAddr].rows
    city_name_id = PClean.resolve_dot_expression(trace.model, :BusinessAddr, :legal_name)
    function is_spirit(pair)
        row = last(pair)
        row[city_name_id] == "SPIRIT PHYSICIAN SERVICES INC"
    end
    filter(is_spirit, rows)
end

function attribute_extractors(model::PClean.PCleanModel)
    physician_attributes = Dict(
        "npi" => PClean.resolve_dot_expression(model, :Obs, :(p.npi)),
        "first" => PClean.resolve_dot_expression(model, :Obs, :(p.first)),
        "last" => PClean.resolve_dot_expression(model, :Obs, :(p.last)),
        "degree" => PClean.resolve_dot_expression(model, :Obs, :(p.degree)),
        "speciality" => PClean.resolve_dot_expression(model, :Obs, :(p.specialty)),
        "school" => PClean.resolve_dot_expression(model, :Obs, :(p.school.name)),
    )

    business_attributes = Dict(
        "legal_name" => PClean.resolve_dot_expression(model, :Obs, :(a.legal_name)),
        "addr" => PClean.resolve_dot_expression(model, :Obs, :(a.addr)),
        "addr2" => PClean.resolve_dot_expression(model, :Obs, :(a.addr2)),
        "zip" => PClean.resolve_dot_expression(model, :Obs, :(a.zip)),
        "city" => PClean.resolve_dot_expression(model, :Obs, :(a.city.name)),
    )

    function attributes(row)
        physician_attr =
            Dict(attribute => row[id] for (attribute, id) in physician_attributes)
        business_attr =
            Dict(attribute => row[id] for (attribute, id) in business_attributes)
        physician_id = row[PClean.resolve_dot_expression(model, :Obs, :p)]
        business_id = row[PClean.resolve_dot_expression(model, :Obs, :a)]
        return (physician_id, business_id), physician_attr, business_attr
    end

    return attributes
end

function build_response(samples)
    p_hist, a_hist = histograms(samples)
    data = unique(x -> x[1], samples)
    data, p_hist, a_hist
end

function publish_tables(possibilities::Dict, trace::Dict{Symbol, PClean.TableTrace})
    serialize("server/physician.jls", trace)
    serialize("server/possibilities.jls", possibilities)
    return
end
