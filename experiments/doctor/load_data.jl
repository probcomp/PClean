using CSV
using DataFrames
all_data = CSV.File("datasets/physician_full.csv") |> DataFrame
possibilities = Dict(col => unique(skipmissing(all_data[!, col]))
                     for col in names(all_data))

# Get cities per City2Zip3
all_data = all_data[:, :]
sort!(all_data, [Symbol("Medical school name"), :Credential])

all_data[:, :City2Zip3] = String[map(x -> "$(x[:City][1:2])-$(x[Symbol("Zip Code")][1:3])", eachrow(all_data))...]
cities = Dict{String, Set{String}}(c => Set{String}() for c in unique(all_data[!,:City2Zip3]))
for r in eachrow(all_data)
 push!(cities[r[:City2Zip3]], r[:City])
end
cities = Dict{String, Vector{String}}(c => String[cities[c]...] for c in keys(cities))
all_data = map(x -> begin
  if ismissing(x[:Credential])
    x[:Credential] = ""
  end
  if ismissing(x[Symbol("Line 2 Street Address")])
    x[Symbol("Line 2 Street Address")] = ""
  end
  if ismissing(x[Symbol("Organization legal name")])
    x[Symbol("Organization legal name")] = ""
  end
  return x
end, eachrow(all_data)) |> DataFrame
