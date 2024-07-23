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
const FIRSTNAMES = possibilities["First Name"]
const LASTNAMES = possibilities["Last Name"]
const ADDRS = possibilities["Line 1 Street Address"]
const ADDRS2 = possibilities["Line 2 Street Address"]
const CITIES = possibilities["City"]
const ZIPS = possibilities["Zip Code"]

PClean.@model PhysicianModel begin
  @class School begin
    @learned school_proportions::ProportionsParameter
    name ~ ChooseProportionally(SCHOOLS, school_proportions); #@guaranteed name
  end

  @class Physician begin
    @learned error_prob::ProbParameter{1.0, 1000.0}
    @learned degree_proportions::Dict{String, ProportionsParameter{3.0}}
    @learned specialty_proportions::Dict{String, ProportionsParameter{3.0}}
    @learned first_name_proportions::ProportionsParameter{3.0}
    @learned last_name_proportions::ProportionsParameter{3.0}

    npi ~ NumberCodePrior(); #@guaranteed npi
    first ~ ChooseProportionally(FIRSTNAMES, first_name_proportions)
    last ~ ChooseProportionally(LASTNAMES, last_name_proportions)
    school ~ School
    begin
      degree ~ ChooseProportionally(CREDENTIALS, degree_proportions[school.name])
      specialty ~ ChooseProportionally(SPECIALITIES, specialty_proportions[degree])
      degree_obs ~ MaybeSwap(degree, CREDENTIALS, error_prob)
    end
  end

  @class City begin
    @learned city_proportions::ProportionsParameter{3.0}
    name ~ ChooseProportionally(CITIES, city_proportions)
  end

  @class BusinessAddr begin
    @learned addr_proportions::ProportionsParameter{3.0}
    @learned addr2_proportions::ProportionsParameter{3.0}
    @learned legal_name_proportions::ProportionsParameter{3.0}
    @learned zip_proportions::ProportionsParameter{3.0}
    addr ~ ChooseProportionally(ADDRS, addr_proportions)
    addr2 ~ ChooseProportionally(ADDRS2, addr2_proportions)
    zip ~ ChooseProportionally(ZIPS, zip_proportions)
    legal_name ~ ChooseProportionally(BUSINESSES, legal_name_proportions)

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

include("utilities.jl")
existing_physicians = keys(trace.tables[:Physician].rows)
existing_businesses = keys(trace.tables[:BusinessAddr].rows) 
existing_observations = Set([(row[PClean.resolve_dot_expression(trace.model,:Obs, :p)], row[PClean.resolve_dot_expression(trace.model, :Obs, :a)]) for (id, row) in trace.tables[:Obs].rows])


# gilmans = find_person(trace,firstname="STEVEN", lastname="GILMAN")
# spirit_service_instances = find_spirit_service(trace)

# INFERENCE CONTINUED
table = deserialize("results/physician.jls")
trace = PClean.PCleanTrace(PhysicianModel, table);

row_id = rand(10000:20000)
row_trace = Dict{PClean.VertexID, Any}()
row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(p.first))] = "STEVEN"
row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(p.last))] = "GILMAN"
# row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(a.city.c2z3))] = "CA-170"
# row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(a.addr))] = "429 N 21ST ST"
# row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(a.addr2))] = ""
row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(a.legal_name))] = "SPIRIT PHYSICIAN SERVICES INC"
# row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(a.zip))] = String15("170112202")
# row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(p.degree))] = "MD"
# row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(p.degree_obs))] = "MD"
# row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(p.specialty))] = "DIAGNOSTIC RADIOLOGY"

obs = trace.tables[:Obs].observations
obs[row_id] = row_trace

extractor = attribute_extractors(PhysicianModel)
# results = filter(x->x[1] in existing_observations, extractor.(samples))
samples = []
for _ in 1:1000
  PClean.run_smc!(trace, :Obs, row_id, PClean.InferenceConfig(40,5))
  r_ = copy(trace.tables[:Obs].rows[row_id])
  # println(extractor(r_))
  # println(r_[])
  info = extractor(r_)
  if info[1] in existing_observations
    push!(samples, extractor(r_))
  end
end
# gilmans = find_person(trace,firstname="STEVEN", lastname="GILMAN")
# PClean.save_results("results", "physician", trace, observations)

# [row[PClean.resolve_dot_expression(trace.model, :Physician, :last)] for (id, row) in find_person(trace,firstname="STEVEN")]



function histograms(results)
  physicians = Dict{Symbol, Int}()
  businesses = Dict{Symbol, Int}()
  for r in results
    physician_id = first(r[1])
    if !(physician_id in keys(physicians))
      physicians[physician_id] = 0
    end
    physicians[physician_id]+=1

    business_id = last(r[1])
    if !(business_id in keys(businesses))
      businesses[business_id] = 0
    end
    businesses[business_id]+=1
  end
  physicians, businesses
end

histograms(samples)
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


