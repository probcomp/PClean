using PClean

include("load_data.jl")

units = [Transformation(identity, identity, x -> 1.0),
        Transformation(x -> x/1000.0, x -> x*1000.0, x -> 1/1000.0)]

PClean.@model RentsModel begin
  @class County begin
    @learned state_pops::ProportionsParameter
    countykey ~ Unmodeled()
    @guaranteed countykey
    name ~ StringPrior(10, 35, possibilities[countykey])
    state ~ ChooseProportionally(states, state_pops)
  end;

  @class Obs begin
    @learned avg_rent::Dict{String, MeanParameter{1500, 1000}}
    county ~ County
    county_name ~ AddTypos(county.name, 2)
    br ~ ChooseUniformly(room_types)
    unit ~ ChooseUniformly(units)
    rent_base = avg_rent["$(county.state)_$(county.countykey)_$(br)"]
    rent ~ TransformedGaussian(rent_base, 150.0, unit)
    corrected = round(unit.backward(rent))
  end;
end;

query = @query RentsModel.Obs [
  CountyKey county.countykey
  County county.name county_name
  State county.state
  "Room Type" br
  "Monthly Rent" corrected rent
];

config = PClean.InferenceConfig(1, 2; use_mh_instead_of_pg=true, rejuv_frequency=500)
observations = [ObservedDataset(query, dirty_table)]

@time begin
  trace = initialize_trace(observations, config);
  run_inference!(trace, config);
end

PClean.save_results("results", "rent", trace, observations)
println(evaluate_accuracy(dirty_table, clean_table, trace.tables[:Obs], query))

row_trace = Dict{PClean.VertexID, Any}()
observed_county_name_key = PClean.resolve_dot_expression(trace.model, :Obs, :county_name)
row_trace[observed_county_name_key] = "New Haven County"
rent_key = PClean.resolve_dot_expression(trace.model, :Obs, :rent)
row_trace[rent_key] = 1000.0
county_key = PClean.resolve_dot_expression(trace.model, :Obs, :(county.countykey))
row_trace[county_key] = "Nw"

obs = trace.tables[:Obs].observations
row_id = rand(10000:20000)
obs[row_id] = row_trace

samples = []
br_idx = PClean.resolve_dot_expression(trace.model, :Obs, :br)
for _ in 1:100
  PClean.run_smc!(trace, :Obs, row_id, PClean.InferenceConfig(1, 10))
  push!(samples, trace.tables[:Obs].rows[row_id][br_idx])
end

count = Dict{String, Int}()
for s in samples
  count[s] = get(count, s, 0)+1
end 
l = collect(count)
l[partialsortperm(l, 1:3, by=last, rev=true)]
