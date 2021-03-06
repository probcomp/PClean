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

println(evaluate_accuracy(dirty_table, clean_table, trace.tables[:Obs], query))