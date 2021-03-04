using CSV
using DataFrames: DataFrame

dataset = "hospital"
dirty_table = CSV.File("datasets/$dataset.csv") |> DataFrame
clean_table_long = CSV.File("datasets/$(dataset)_clean.csv") |> DataFrame
clean_table = unstack(clean_table_long, :tid, :attribute, :correct_val)

clean_table[!, :PhoneNumber] = map(x -> "$x", clean_table[!, :PhoneNumber])
clean_table[!, :ZipCode] = map(x -> "$x", clean_table[!, :ZipCode])
clean_table[!, :ProviderNumber] = map(x -> "$x", clean_table[!, :ProviderNumber])

possibilities = Dict(col => remove_missing(unique(collect(dirty_table[!, col])))
                     for col in names(dirty_table))
