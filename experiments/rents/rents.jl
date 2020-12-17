using PClean

include("load_data.jl")

units = [Transformation(identity, identity, x -> 1.0),
        Transformation(x -> x/1000.0, x -> x*1000.0, x -> 1/1000.0)]

PClean.@pcleanmodel RentsModel begin
  @class County begin
    @learned state_pops::ProportionsParameter
    col1 ~ Unmodeled()
    @guaranteed col1
    name ~ StringPrior(10, 35, possibilities[col1])
    state ~ ChooseProportionally(states, state_pops)
  end;

  @class Obs begin
    @learned avg_rent::Dict{String, MeanParameter{1500, 1000}}
    begin
      county ~ County
      county_name ~ AddTypos(county.name, 2)
      br ~ ChooseUniformly(room_types)
      unit ~ ChooseUniformly(units)
      rent_base = avg_rent["$(county.state)_$(county.col1)_$(br)"]
      rent ~ TransformedGaussian(rent_base, 150.0, unit)
    end
    corrected = round(unit.backward(rent))
  end;
end;

query = @query RentsModel.Obs [
  Column1 county.col1
  County county.name county_name
  State county.state
  "Room Type" br
  "Monthly Rent" corrected rent
];

config = PClean.InferenceConfig(1, 2; use_mh_instead_of_pg=true, rejuv_frequency=500)
observations = [ObservedDataset(query, table)]

@time begin
  trace = initialize_trace(observations, config);
  run_inference!(trace, config);
end

@time run_inference!(tr, config)
println(evaluate_accuracy(table, clean_table, trace.tables[:Obs], query))