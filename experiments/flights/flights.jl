using PClean

include("load_data.jl")

websites = unique(dirty_table.src)

PClean.@pcleanmodel FlightsModel begin

  @class TrackingWebsite begin
    name ~ StringPrior(2, 30, websites)
  end

  @class Flight begin
    flight_id ~ StringPrior(10, 20, flight_ids); @guaranteed flight_id
    sdt ~ TimePrior(times_for_flight["$flight_id-sched_dep_time"])
    sat ~ TimePrior(times_for_flight["$flight_id-sched_arr_time"])
    adt ~ TimePrior(times_for_flight["$flight_id-act_dep_time"])
    aat ~ TimePrior(times_for_flight["$flight_id-act_arr_time"])
  end

  @class Obs begin
    @learned error_probs::Dict{String, ProbParameter{10.0, 50.0}}
    flight ~ Flight; src ~ TrackingWebsite
    error_prob = lowercase(src.name) == lowercase(flight.flight_id[1:2]) ? 1e-5 : error_probs[src.name]
    sdt ~ MaybeSwap(flight.sdt, times_for_flight["$(flight.flight_id)-sched_dep_time"], error_prob)
    sat ~ MaybeSwap(flight.sat, times_for_flight["$(flight.flight_id)-sched_arr_time"], error_prob)
    adt ~ MaybeSwap(flight.adt, times_for_flight["$(flight.flight_id)-act_dep_time"],   error_prob)
    aat ~ MaybeSwap(flight.aat, times_for_flight["$(flight.flight_id)-act_arr_time"],   error_prob)
  end

end;

query = @query FlightsModel.Obs [
  sched_dep_time flight.sdt sdt
  sched_arr_time flight.sat sat
  act_dep_time flight.adt adt
  act_arr_time flight.aat aat
  flight flight.flight_id
  src src.name
];

observations = [ObservedDataset(query, dirty_table)]
config = PClean.InferenceConfig(5, 2; use_mh_instead_of_pg=true)
@time begin 
  tr = initialize_trace(observations, config);
  run_inference!(tr, config)
end

println(evaluate_accuracy(dirty_table, clean_data, tr.tables[:Obs], query))
