using CSV

dataset = "hospital"
dirty_table = CSV.File("datasets/$dataset.csv") |> CSV.DataFrame!
clean_table = CSV.File("datasets/clean_hospital.csv") |> CSV.DataFrame!

clean_table[!, :PhoneNumber] = map(x -> "$x", clean_table.PhoneNumber)
clean_table[!, :ZipCode] = map(x -> "$x", clean_table.ZipCode)
clean_table[!, :ProviderNumber] = map(x -> "$x", clean_table.ProviderNumber)

possibilities = Dict(col => remove_missing(unique(collect(dirty_table[!, col])))
                     for col in names(dirty_table))
