using PClean

include("load_data.jl")

PClean.@model CityModel begin 
    @class Country begin
        @learned state_proportions::ProportionsParameter
        country ~ StringPrior(3, 30, possibilities[:country])
    end
    @class City begin
        country ~ Country
        name ~ StringPrior(3, 30, possibilities[:item])
        population ~ ChooseUniformly(possibilities[:population])
    end;
    @class Record begin
        city ~ City
        name ~ AddTypos(city.name)
        country ~ AddTypos(city.country.country)
    end;
end;

query = @query CityModel.Record [
    item city.name;
    country city.country.country;
    population city.population;
];

config = PClean.InferenceConfig(1, 2; use_mh_instead_of_pg=true);
observations = [ObservedDataset(query, clean_table)];
@time begin 
    trace = initialize_trace(observations, config);
    run_inference!(trace, config);
end

# results = evaluate_accuracy(dirty_table, clean_table, trace.tables[:Record], query)
# PClean.save_results("results", "hospital", trace, observations)
# println(results)
