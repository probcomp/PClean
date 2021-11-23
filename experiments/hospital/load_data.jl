using CSV
using DataFrames: DataFrame

dataset = "hospital"
dirty_table = CSV.File("datasets/$(dataset)_dirty.csv", stringtype=String) |> DataFrame
clean_table = CSV.File("datasets/$(dataset)_clean.csv", stringtype=String) |> DataFrame

# In the dirty data, CSV.jl infers that PhoneNumber, ZipCode, and ProviderNumber
# are strings. Our PClean script also models these columns as string-valued.
# However, in the clean CSV file (without typos) it infers they are
# numbers. To facilitate comparison of PClean's results (strings) with 
# ground-truth, we preprocess the clean values to convert them into strings.
clean_table[!, :PhoneNumber] = map(x -> "$x", clean_table[!, :PhoneNumber])
clean_table[!, :ZipCode] = map(x -> "$x", clean_table[!, :ZipCode])
clean_table[!, :ProviderNumber] = map(x -> "$x", clean_table[!, :ProviderNumber])

# Stores sets of unique observed values of each attribute.
possibilities = Dict(col => remove_missing(unique(collect(dirty_table[!, col])))
                     for col in propertynames(dirty_table))
