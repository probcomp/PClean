using CSV
using DataFrames: DataFrame

# Load data
dataset = "rents"
dirty_table = CSV.File("datasets/$(dataset)_dirty.csv", stringtype=String) |> DataFrame
clean_table = CSV.File("datasets/$(dataset)_clean.csv", stringtype=String) |> DataFrame

dirty_table[!, :CountyKey] = map(x -> "$(x[1])$(split(x)[1][end])", dirty_table[!, :County])

possibilities = Dict(c => Set() for c in unique(dirty_table.CountyKey))
for r in eachrow(dirty_table)
  push!(possibilities[r[:CountyKey]], r[:County])
end
possibilities = Dict(c => [possibilities[c]...] for c in keys(possibilities))

const states = unique(filter(x -> !ismissing(x), dirty_table.State))
const room_types = ["studio", "1br", "2br", "3br", "4br"]

