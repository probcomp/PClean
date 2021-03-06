using CSV
using DataFrames: DataFrame

dataset = "hospital"
dirty_table = CSV.File("datasets/$(dataset)_dirty.csv") |> DataFrame
clean_table = CSV.File("datasets/$(dataset)_clean.csv") |> DataFrame

clean_table[!, :PhoneNumber] = map(x -> "$x", clean_table[!, :PhoneNumber])
clean_table[!, :ZipCode] = map(x -> "$x", clean_table[!, :ZipCode])
clean_table[!, :ProviderNumber] = map(x -> "$x", clean_table[!, :ProviderNumber])

possibilities = Dict(col => remove_missing(unique(collect(dirty_table[!, col])))
                     for col in propertynames(dirty_table))