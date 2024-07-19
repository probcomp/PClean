using CSV
using DataFrames: DataFrame
using DataFrames

dataset = "cities"
# dirty_table = CSV.File("datasets/$(dataset)_dirty.csv") |> DataFrame
clean_table = CSV.File("datasets/$(dataset)_clean.csv") |> DataFrame

# Rename the columns. Use only the "value" columns.
col_names = filter(name->endswith(name, ".value"), names(clean_table))
new_table = select(clean_table, col_names)
rename!(new_table, "item.value" => "qidLabel.value")
col_names = names(new_table)
new_col_names = map(name->name[1:end-11], names(new_table))
rename!(new_table, col_names .=> new_col_names)

CSV.write("datasets/cities_table.csv", new_table)
clean_table = new_table

clean_table = first(clean_table, 400)
clean_table = clean_table[!, [:item, :country, :population]]
# In the dirty data, CSV.jl infers that PhoneNumber, ZipCode, and ProviderNumber
# are strings. Our PClean script also models these columns as string-valued.
# However, in the clean CSV file (without typos) it infers they are
# numbers. To facilitate comparison of PClean's results (strings) with 
# ground-truth, we preprocess the clean values to convert them into strings.
# clean_table[!, :PhoneNumber] = map(x -> "$x", clean_table[!, :PhoneNumber])
# clean_table[!, :ZipCode] = map(x -> "$x", clean_table[!, :ZipCode])
# clean_table[!, :ProviderNumber] = map(x -> "$x", clean_table[!, :ProviderNumber])

# Stores sets of unique observed values of each attribute.
possibilities = Dict(col => remove_missing(unique(collect(clean_table[!, col])))
                     for col in propertynames(clean_table))