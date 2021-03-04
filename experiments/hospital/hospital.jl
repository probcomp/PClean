using PClean

include("load_data.jl")

PClean.@model HospitalModel begin 
    @class County begin
        @learned state_proportions::ProportionsParameter
        state ~ ChooseProportionally(possibilities[:State], state_proportions)
        county ~ StringPrior(3, 30, possibilities[:CountyName])
    end;
    @class Place begin
        county ~ County
        city ~ StringPrior(3, 30, possibilities[:City])
    end;
    @class Condition begin
        desc ~ StringPrior(5, 35, possibilities[:Condition])
    end;
    @class Measure begin
        code ~ ChooseUniformly(possibilities[:MeasureCode])
        name ~ ChooseUniformly(possibilities[:MeasureName])
        condition ~ Condition
    end;
    @class HospitalType begin
        desc ~ StringPrior(10, 30, possibilities[:HospitalType])
    end;
    @class Hospital begin
        @learned owner_dist::ProportionsParameter
        @learned service_dist::ProportionsParameter
        loc ~ Place        
        type ~ HospitalType
        provider ~ ChooseUniformly(possibilities[:ProviderNumber])
        name ~ StringPrior(3, 50, possibilities[:HospitalName])
        addr ~ StringPrior(10, 30, possibilities[:Address1])
        phone ~ StringPrior(10, 10, possibilities[:PhoneNumber])
        owner ~ ChooseProportionally(possibilities[:HospitalOwner], owner_dist)
        zip ~ ChooseUniformly(possibilities[:ZipCode])
        service ~ ChooseProportionally(possibilities[:EmergencyService], service_dist)
    end;
    @class Obs begin
        begin
            hosp     ~ Hospital;                         service ~ AddTypos(hosp.service)
            provider ~ AddTypos(hosp.provider);          name    ~ AddTypos(hosp.name)
            addr     ~ AddTypos(hosp.addr);              city    ~ AddTypos(hosp.loc.city)
            state    ~ AddTypos(hosp.loc.county.state);  zip     ~ AddTypos(hosp.zip)
            county   ~ AddTypos(hosp.loc.county.county); phone   ~ AddTypos(hosp.phone)
            type     ~ AddTypos(hosp.type.desc);         owner   ~ AddTypos(hosp.owner)
        end
        begin
            metric ~ Measure
            code ~ AddTypos(metric.code);
            mname ~ AddTypos(metric.name);
            condition ~ AddTypos(metric.condition.desc)
            stateavg = "$(hosp.loc.county.state)_$(metric.code)"
            stateavg_obs ~ AddTypos(stateavg)
        end
    end;
end;

query = @query HospitalModel.Obs [
    ProviderNumber   hosp.provider          provider
    HospitalName     hosp.name              name
    HospitalType     hosp.type.desc         type
    HospitalOwner    hosp.owner             owner
    Address1         hosp.addr              addr
    PhoneNumber      hosp.phone             phone
    EmergencyService hosp.service           service
    City             hosp.loc.city          city
    CountyName       hosp.loc.county.county county
    State            hosp.loc.county.state  state
    ZipCode          hosp.zip               zip
    Condition        metric.condition.desc  condition
    MeasureCode      metric.code            code
    MeasureName      metric.name            mname
    Stateavg         stateavg               stateavg_obs
];

config = PClean.InferenceConfig(1, 2; use_mh_instead_of_pg=true, reporting_frequency=1);
observations = [ObservedDataset(query, dirty_table)];
@time begin 
    trace = initialize_trace(observations, config);
    run_inference!(trace, config);
end

results = evaluate_accuracy(dirty_table, clean_table, trace.tables[:Obs], query)
PClean.save_results("results", "hospital_badmodel", trace, observations)
println(results)
