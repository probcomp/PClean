using PClean
using CSV
using DataFrames: DataFrame, describe

function trace_to_dfs(trace)
    dfs = Dict{Symbol,DataFrame}()
    for (class, table) in trace.tables 
        println(class)
        dfs[class] = DataFrame(;
            id = collect(keys(table.rows)),
            [k => [row[i] for row in values(table.rows)]
                for (k, i) in trace.model.classes[class].names
                if !occursin("#", string(k)) && PClean.is_saveable(trace.model.classes[class].nodes[i])]...)
    end
    return dfs
end

df = CSV.File("datasets/linkage/dataset4a.csv", delim=", ") |> DataFrame
df = df[!, [:given_name, :surname]]
display(describe(df))

all_given_names = Vector{String}(unique(vcat(
    filter((v) -> !ismissing(v), df[!, :given_name]))))

all_surnames = Vector{String}(unique(vcat(
    filter((v) -> !ismissing(v), df[!, :surname]))))

println("number of unique given names in the data set: $(length(all_given_names))")
println("number of unique surnames in the data set: $(length(all_surnames))")

PClean.@model CustomerModel begin

    @class FirstNames begin
        name ~ StringPrior(1, 60, all_given_names)
    end

    @class LastNames begin
        name ~ StringPrior(1, 60, all_surnames)
    end

    @class Person begin
        given_name ~ FirstNames
        surname ~ LastNames
    end;

    @class Obs begin
        begin
            person ~ Person
            given_name ~ AddTypos(person.given_name.name)
            surname ~ AddTypos(person.surname.name)
        end
    end;

end;

query = @query CustomerModel.Obs [
    given_name person.given_name.name given_name
    surname person.surname.name surname
];

observations = [ObservedDataset(query, df)]
config = PClean.InferenceConfig(5, 2; use_mh_instead_of_pg=true)
@time begin 
    tr = initialize_trace(observations, config);
    run_inference!(tr, config)
end

dfs = trace_to_dfs(tr)
println(dfs[:Person])
n = size(dfs[:Person])[1]
println("inferred $n unique rows in the table Person")
