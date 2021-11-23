using CSV
using DataFrames: DataFrame

dataset = "flights"
dirty_table = CSV.File("datasets/$(dataset)_dirty.csv", stringtype=String) |> DataFrame
clean_data = CSV.File("datasets/$(dataset)_clean.csv", stringtype=String) |> DataFrame

times_for_flight = Dict{String, Set{String}}("$fl-$field" => Set() for fl in unique(dirty_table.flight) for field in [:sched_dep_time, :sched_arr_time, :act_dep_time, :act_arr_time])
for row in eachrow(dirty_table)
  for field  in [:sched_dep_time, :sched_arr_time, :act_dep_time, :act_arr_time]
    !ismissing(row[field]) && push!(times_for_flight["$(row[:flight])-$field"], row[field])
  end
end

times_for_flight = Dict(fl => [unique(times_for_flight[fl])...] for fl in keys(times_for_flight))
flight_ids = unique(dirty_table.flight)
