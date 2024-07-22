using PClean
using Serialization

include("load_data.jl")

##############
# PHYSICIANS #
##############
const SPECIALITIES = possibilities["Primary specialty"]
const CREDENTIALS = possibilities["Credential"]
const SCHOOLS = possibilities["Medical school name"]
const BUSINESSES = possibilities["Organization legal name"]
const LASTNAMES = possibilities["Last Name"]
const ADDRS = possibilities["Line 1 Street Address"]

PClean.@model PhysicianModel begin
  @class School begin
    @learned school_proportions::ProportionsParameter
    name ~ ChooseProportionally(SCHOOLS, school_proportions); #@guaranteed name
  end

  @class Physician begin
    @learned error_prob::ProbParameter{1.0, 1000.0}
    @learned degree_proportions::Dict{String, ProportionsParameter{3.0}}
    @learned specialty_proportions::Dict{String, ProportionsParameter{3.0}}
    npi ~ NumberCodePrior(); #@guaranteed npi
    first ~ Unmodeled()
    # last ~ Unmodeled()
    @learned last_name_proportions::ProportionsParameter{3.0}
    last ~ ChooseProportionally(LASTNAMES, last_name_proportions)
    school ~ School
    begin
      degree ~ ChooseProportionally(CREDENTIALS, degree_proportions[school.name])
      specialty ~ ChooseProportionally(SPECIALITIES, specialty_proportions[degree])
      degree_obs ~ MaybeSwap(degree, CREDENTIALS, error_prob)
    end
  end

  @class City begin
    c2z3 ~ Unmodeled(); #@guaranteed c2z3
    name ~ StringPrior(3, 30, cities[c2z3])
  end

  @class BusinessAddr begin
    @learned addr_proportions::ProportionsParameter{3.0}
    addr ~ ChooseProportionally(ADDRS, addr_proportions)
    # addr ~ Unmodeled(); #@guaranteed addr

    addr2 ~ Unmodeled(); #@guaranteed addr2
    zip ~ StringPrior(3, 10, String[]); #@guaranteed zip

    legal_name ~ Unmodeled(); #@guaranteed legal_name
    begin
      city ~ City
      city_name ~ AddTypos(city.name, 2)
    end
  end

  @class Obs begin
    p ~ Physician
    a ~ BusinessAddr
  end
end

query = @query PhysicianModel.Obs [
  "NPI" p.npi
  "Primary specialty" p.specialty
  "First Name" p.first
  "Last Name" p.last
  "Medical school name" p.school.name
  "Credential" p.degree p.degree_obs
  "City2Zip3" a.city.c2z3
  "City" a.city.name a.city_name
  "Line 1 Street Address" a.addr
  "Line 2 Street Address" a.addr2
  "Zip Code" a.zip
  "Organization legal name" a.legal_name
];

observations = [ObservedDataset(query, all_data[1:1000,:])]
config = PClean.InferenceConfig(5, 5; use_mh_instead_of_pg=true)

@time begin 
  trace = initialize_trace(observations, config);
  run_inference!(trace, config)
end

function countmap(samples::Vector{T}) where {T}
  count = Dict{T, Int}()
  for s in samples
    count[s] = get(count, s, 0)+1
  end 
  count
end

serialize("results/physician.jls", trace.tables)
table = deserialize("results/physician.jls")
trace = PClean.PCleanTrace(PhysicianModel, table);
# trace = PClean.PCleanTrace(PhysicianModel, table_);

PClean.save_results("results", "physician", trace, observations)

existing_physicians = keys(trace.tables[:Physician].rows)
existing_businesses = keys(trace.tables[:BusinessAddr].rows) 
existing_observations = Set([(row[PClean.resolve_dot_expression(trace.model,:Obs, :p)], row[PClean.resolve_dot_expression(trace.model, :Obs, :a)]) for (id, row) in trace.tables[:Obs].rows])

# gilman = filter(pair-> (row = last(pair); row[5] == String31("STEVEN") && row[6] == String31("GILMAN")), trace.tables[:Physician].rows)
function find_person(trace; firstname=nothing, lastname=nothing)
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

gilmans = find_person(trace,firstname="STEVEN", lastname="GILMAN")
# [row[PClean.resolve_dot_expression(trace.model, :Physician, :last)] for (id, row) in find_person(trace,firstname="STEVEN")]

function find_spirit_service(trace)
  rows = trace.tables[:BusinessAddr].rows
  city_name_id = PClean.resolve_dot_expression(trace.model, :BusinessAddr, :legal_name)
  function is_spirit(pair)
    row = last(pair) 
    row[city_name_id] == "SPIRIT PHYSICIAN SERVICES INC"
  end
  filter(is_spirit, rows)
end
spirit_service_instances = find_spirit_service(trace)
# INFERENCE CONTINUED
row_id = rand(10000:20000)

row_trace = Dict{PClean.VertexID, Any}()
row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(p.first))] = "STEVEN"
# row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(p.last))] = "GILMAN"
# row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(p.school.name))] = "ALBANY MEDICAL COLLEGE OF UNION UNIVERSITY"
row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(a.city.c2z3))] = "CA-170"
# row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(a.addr))] = "429 N 21ST ST"
row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(a.addr2))] = ""
row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(a.legal_name))] = "SPIRIT PHYSICIAN SERVICES INC"
row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(a.zip))] = String15("170112202")
# row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(p.degree))] = "MD"
# row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(p.degree_obs))] = "MD"
row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(p.specialty))] = "DIAGNOSTIC RADIOLOGY"

obs = trace.tables[:Obs].observations
obs[row_id] = row_trace

samples = Dict{Int64,Any}[]
for _ in 1:10
  PClean.run_smc!(trace, :Obs, row_id, PClean.InferenceConfig(40,5))
  push!(samples, copy(trace.tables[:Obs].rows[row_id]))
  # temp = find_spirit_service(trace)
  # display(temp)
  # business_addr_ids_different = symdiff(keys(spirit_service_instances), keys(temp))
  # println(business_addr_ids_different)
  # println(trace.tables[:Obs].rows[row_id][PClean.resolve_dot_expression(trace.model, :Obs, :a)])
  println(trace.tables[:Obs].rows[row_id][PClean.resolve_dot_expression(trace.model, :Obs, :a)])
  println(trace.tables[:Obs].rows[row_id][PClean.resolve_dot_expression(trace.model, :Obs, :(a.legal_name))])
  println(trace.tables[:Obs].rows[row_id][PClean.resolve_dot_expression(trace.model, :Obs, :(a.addr))])
  println(trace.tables[:Obs].rows[row_id][PClean.resolve_dot_expression(trace.model, :Obs, :(a.city.name))])
  println(trace.tables[:Obs].rows[row_id][PClean.resolve_dot_expression(trace.model, :Obs, :p)])
  println(trace.tables[:Obs].rows[row_id][PClean.resolve_dot_expression(trace.model, :Obs, :(p.first))])
  println(trace.tables[:Obs].rows[row_id][PClean.resolve_dot_expression(trace.model, :Obs, :(p.last))])
  println(trace.tables[:Obs].rows[row_id][PClean.resolve_dot_expression(trace.model, :Obs, :(p.school.name))])
  println()
  # push!(last_name_samples, row[PClean.resolve_dot_expression(trace.model, :Obs, :(p.last))])
end
gilmans = find_person(trace,firstname="STEVEN", lastname="GILMAN")
PClean.save_results("results", "physician", trace, observations)

[row[PClean.resolve_dot_expression(trace.model, :Physician, :last)] for (id, row) in find_person(trace,firstname="STEVEN")]

function attribute_extractors(model::PClean.PCleanModel)
    physician_attributes = Dict(
      "npi"=>PClean.resolve_dot_expression(model, :Obs, :(p.npi)),
      "first"=>PClean.resolve_dot_expression(model, :Obs, :(p.first)),
      "last"=>PClean.resolve_dot_expression(model, :Obs, :(p.last)),
      "degree"=>PClean.resolve_dot_expression(model, :Obs, :(p.degree)),
      "speciality"=>PClean.resolve_dot_expression(model, :Obs, :(p.specialty)),
      "school" => PClean.resolve_dot_expression(model, :Obs, :(p.school.name))
    )

    business_attributes = Dict(
      "legal_name" => PClean.resolve_dot_expression(model, :Obs, :(a.legal_name)),
      "addr" => PClean.resolve_dot_expression(model, :Obs, :(a.addr)),
      "addr2" => PClean.resolve_dot_expression(model, :Obs, :(a.addr2)),
      "zip" => PClean.resolve_dot_expression(model, :Obs, :(a.zip)),
      "city" => PClean.resolve_dot_expression(model, :Obs, :(a.city.name)),
    )

    function attributes(row)
      physician_attr = Dict(attribute=>row[id] for (attribute, id) in physician_attributes)
      business_attr = Dict(attribute=>row[id] for (attribute, id) in business_attributes)
      physician_id = row[PClean.resolve_dot_expression(model, :Obs, :p)]
      business_id = row[PClean.resolve_dot_expression(model, :Obs, :a)]
      return (physician_id, business_id), physician_attr, business_attr
    end

    return attributes
end

extractor = attribute_extractors(PhysicianModel)
# sum(extractor(s)[1] in existing_physicians for s in samples)
extractor(samples[5])
ob = extractor(samples[5])[1]
ob in existing_observations
# extractor(samples[1])[1] in existing_physicians

# p_freq = countmap(physician_ids)
# gilman in keys(p_freq)

# specialty_freq = countmap(specialty_samples);
# l  = collect(specialty_freq);
# l[partialsortperm(l, 1:3, by=last, rev=true)]

# last_freq = countmap(last_name_samples);
# l  = collect(last_freq);
# l[partialsortperm(l, 1:3, by=last, rev=true)]

# println(trace.tables[:Obs].rows[row_id])

# trace.tables[:Obs].rows[row_id][1] # physician foreign key
# trace.tables[:Obs].rows[row_id][8] # school foreign key
# trace.tables[:Obs].rows[row_id][18] # business foreign key
# trace.tables[:Obs].rows[row_id][26] # city foreign key

# PClean.resolve_dot_expression(trace.model, :Obs, :(p))
# PClean.resolve_dot_expression(trace.model, :Obs, :(p.school))
# PClean.resolve_dot_expression(trace.model, :Obs, :a)
# PClean.resolve_dot_expression(trace.model, :Obs, :(a.city))


