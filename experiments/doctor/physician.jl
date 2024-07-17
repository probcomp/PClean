using PClean

include("load_data.jl")

##############
# PHYSICIANS #
##############
const SPECIALITIES = possibilities["Primary specialty"]
const CREDENTIALS = possibilities["Credential"]

PClean.@model PhysicianModel begin
  @class School begin
    name ~ Unmodeled(); #@guaranteed name
  end

  @class Physician begin
    @learned error_prob::ProbParameter{1.0, 1000.0}
    @learned degree_proportions::Dict{String, ProportionsParameter{3.0}}
    @learned specialty_proportions::Dict{String, ProportionsParameter{3.0}}
    npi ~ NumberCodePrior(); #@guaranteed npi
    first ~ Unmodeled()
    last ~ Unmodeled()
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
    addr ~ Unmodeled(); #@guaranteed addr
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

observations = [ObservedDataset(query, all_data[1:300,:])]
config = PClean.InferenceConfig(3, 2; use_mh_instead_of_pg=true)

@time begin 
  trace = initialize_trace(observations, config);
  run_inference!(trace, config)
end

using Serialization
serialize("results/physician.jls", trace.tables)
table_ = deserialize("results/physician.jls")
trace = PClean.PCleanTrace(PhysicianModel, table_);


PClean.save_results("results", "physician", trace, observations)

# INFERENCE CONTINUED
row_trace = Dict{PClean.VertexID, Any}()
# observed_city_addr = PClean.resolve_dot_expression(trace.model, :Obs, :(a.city))
row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(p.school.name))] = "A T STILL UN, ARIZONA SCHL OF DENT.Y & ORAL HLTH"
row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(p.first))] = "SALLY"
row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(p.last))] = "FODERO"
row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(a.city.c2z3))] = "GR-038"
row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(a.addr))] = "655 PORTSMOUTH AVE"
row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(a.addr2))] = ""
row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(a.legal_name))] = ""
# row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(a.city_name))] = String31("LEXINGTON")
# row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(a.zip))] = String15("405043274")

row_id = gensym()
obs = trace.tables[:Obs].observations
obs[row_id] = row_trace

PClean.run_smc!(trace, :Obs, row_id, PClean.InferenceConfig(1,3))
trace.tables[:Obs].rows[row_id]