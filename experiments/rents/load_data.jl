using CSV

# Load data
table = CSV.File("datasets/dirty_rents.csv") |> CSV.DataFrame!
table[!, :Column1] = map(x -> "$(x[1])$(split(x)[1][end])", table[!, :County])

possibilities = Dict(c => Set() for c in unique(table.Column1))
for r in eachrow(table)
  push!(possibilities[r[:Column1]], r[:County])
end
possibilities = Dict(c => [possibilities[c]...] for c in keys(possibilities))

const states = unique(filter(x -> !ismissing(x), table.State))
const room_types = ["studio", "1br", "2br", "3br", "4br"]

clean_table = CSV.File("datasets/clean_rents.csv") |> CSV.DataFrame!
