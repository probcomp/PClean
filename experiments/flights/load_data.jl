using CSV
using DataFrames: DataFrame

dirty_table = CSV.File("datasets/flights_dirty.csv") |> DataFrame

times_for_flight = Dict{String, Set{String}}("$fl-$field" => Set() for fl in unique(dirty_table.flight) for field in [:sched_dep_time, :sched_arr_time, :act_dep_time, :act_arr_time])
for row in eachrow(dirty_table)
  for field  in [:sched_dep_time, :sched_arr_time, :act_dep_time, :act_arr_time]
    !ismissing(row[field]) && push!(times_for_flight["$(row[:flight])-$field"], row[field])
  end
end

times_for_flight = Dict(fl => [unique(times_for_flight[fl])...] for fl in keys(times_for_flight))
flight_ids = unique(dirty_table.flight)
clean_data = CSV.File("datasets/flights_clean.csv") |> DataFrame
