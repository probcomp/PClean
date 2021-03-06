using PClean
using CSV
using DataFrames: DataFrame, describe

df1 = CSV.File("datasets/linkage/dataset4a.csv", delim=", ") |> DataFrame
df2 = CSV.File("datasets/linkage/dataset4b.csv", delim=", ") |> DataFrame

display(describe(df1))
display(describe(df2))

df1 = df1[:, [:given_name, :surname]]
df2 = df2[:, [:given_name, :surname]]

println(df1[1:10,:])
println(df2[1:10,:])

all_given_names = Vector{String}(unique(vcat(
    filter((v) -> !ismissing(v), df1[!, :given_name]),
    filter((v) -> !ismissing(v), df2[!, :given_name]))))

all_surnames = Vector{String}(unique(vcat(
    filter((v) -> !ismissing(v), df1[!, :surname]),
    filter((v) -> !ismissing(v), df2[!, :surname]))))

PClean.@model CustomerModel begin

    @class Person begin
        given_name ~ StringPrior(2, 30, all_given_names)
        surname ~ StringPrior(2, 30, all_surnames)
    end;

    @class Obs1 begin
        person ~ Person
        given_name ~ AddTypos(person.given_name)
        surname ~ AddTypos(person.given_name)
    end;

    #@class Obs2 begin
        #person ~ Person
        #jgiven_name ~ AddTypos(person.given_name)
        #surname ~ AddTypos(person.given_name)
    #end

end;

query1 = @query CustomerModel.Obs1 [
  given_name person.given_name given_name
  surname person.surname surname
];

#query2 = @query CustomerModel.Obs2 [
  #given_name given_name
  #surname surname
#];

observations = [ObservedDataset(query1, df1)]#, ObservedDataset(query2, df2)]
config = PClean.InferenceConfig(5, 2; use_mh_instead_of_pg=true)
@time begin 
  tr = initialize_trace(observations, config);
  run_inference!(tr, config)
end


#println(evaluate_accuracy(dirty_table, clean_data, tr.tables[:Obs], query))
