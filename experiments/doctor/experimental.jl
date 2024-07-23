using PClean
using Serialization

include("utilities.jl")
include("load_data.jl")

##############
# PHYSICIANS #
##############
const SPECIALITIES = possibilities["Primary specialty"]
const CREDENTIALS = possibilities["Credential"]
const SCHOOLS = possibilities["Medical school name"]
const BUSINESSES = possibilities["Organization legal name"]
const FIRSTNAMES = possibilities["First Name"]
const LASTNAMES = possibilities["Last Name"]
const ADDRS = possibilities["Line 1 Street Address"]
const ADDRS2 = possibilities["Line 2 Street Address"]
const CITIES = possibilities["City"]
const ZIPS = possibilities["Zip Code"]

PClean.@model PhysicianModel begin
    @class School begin
        @learned school_proportions::ProportionsParameter
        name ~ ChooseProportionally(SCHOOLS, school_proportions) #@guaranteed name
    end

    @class Physician begin
        @learned error_prob::ProbParameter{1.0,1000.0}
        @learned degree_proportions::Dict{String,ProportionsParameter{3.0}}
        @learned specialty_proportions::Dict{String,ProportionsParameter{3.0}}
        @learned first_name_proportions::ProportionsParameter{3.0}
        @learned last_name_proportions::ProportionsParameter{3.0}

        npi ~ NumberCodePrior() #@guaranteed npi
        first ~ ChooseProportionally(FIRSTNAMES, first_name_proportions)
        last ~ ChooseProportionally(LASTNAMES, last_name_proportions)
        school ~ School
        begin
            degree ~ ChooseProportionally(CREDENTIALS, degree_proportions[school.name])
            specialty ~ ChooseProportionally(SPECIALITIES, specialty_proportions[degree])
            degree_obs ~ MaybeSwap(degree, CREDENTIALS, error_prob)
        end
    end

    @class City begin
        @learned city_proportions::ProportionsParameter{3.0}
        name ~ ChooseProportionally(CITIES, city_proportions)
    end

    @class BusinessAddr begin
        @learned addr_proportions::ProportionsParameter{3.0}
        @learned addr2_proportions::ProportionsParameter{3.0}
        @learned legal_name_proportions::ProportionsParameter{3.0}
        @learned zip_proportions::ProportionsParameter{3.0}
        addr ~ ChooseProportionally(ADDRS, addr_proportions)
        addr2 ~ ChooseProportionally(ADDRS2, addr2_proportions)
        zip ~ ChooseProportionally(ZIPS, zip_proportions)
        legal_name ~ ChooseProportionally(BUSINESSES, legal_name_proportions)

        begin
            city ~ City
            city_name ~ AddTypos(city.name, 2)
        end
    end

    @class Obs begin
        p ~ Physician
        a ~ BusinessAddr
    end
end


query = @query PhysicianModel.Obs [
    "NPI" p.npi
    "Primary specialty" p.specialty
    "First Name" p.first
    "Last Name" p.last
    "Medical school name" p.school.name
    "Credential" p.degree p.degree_obs
    "City" a.city.name a.city_name
    "Line 1 Street Address" a.addr
    "Line 2 Street Address" a.addr2
    "Zip Code" a.zip
    "Organization legal name" a.legal_name
];

observations = [ObservedDataset(query, all_data[1:1000, :])]
config = PClean.InferenceConfig(5, 5; use_mh_instead_of_pg = true)

@time begin
    trace = initialize_trace(observations, config)
    run_inference!(trace, config)
end

# publish_tables(possibilities, trace.tables)
serialize("results/physician.jls", trace.tables)
table = deserialize("results/physician.jls")
trace = PClean.PCleanTrace(PhysicianModel, table);

PClean.save_results("results", "physician", trace, observations)

existing_physicians = keys(trace.tables[:Physician].rows)
existing_businesses = keys(trace.tables[:BusinessAddr].rows)
existing_observations = Set([
    (
        row[PClean.resolve_dot_expression(trace.model, :Obs, :p)],
        row[PClean.resolve_dot_expression(trace.model, :Obs, :a)],
    ) for (id, row) in trace.tables[:Obs].rows
])


# gilmans = find_person(trace,firstname="STEVEN", lastname="GILMAN")
# spirit_service_instances = find_spirit_service(trace)

# INFERENCE CONTINUED
table = deserialize("results/physician.jls")
trace = PClean.PCleanTrace(PhysicianModel, table);

row_id = rand(10000:20000)
row_trace = Dict{PClean.VertexID,Any}()
row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(p.first))] = "SETH"
row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(p.last))] = "RUCHI"
# row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(p.first))] = "STEVEN"
# row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(p.last))] = "GILMAN"
# row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(a.addr))] = "429 N 21ST ST"
# row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(a.addr))] = "123 EVERETT RD"
row_trace[PClean.resolve_dot_expression(trace.model, :Obs, :(a.legal_name))] = "ST. JOHN'S WELL CHILD AND FAMILY CENTER, INC."
# row_trace[PClean.resolve_dot_expression(
#     trace.model,
#     :Obs,
#     :(a.legal_name),
# )] = "SPIRIT PHYSICIAN SERVICES INC"

obs = trace.tables[:Obs].observations
obs[row_id] = row_trace

extractor = attribute_extractors(PhysicianModel)
# results = filter(x->x[1] in existing_observations, extractor.(samples))
samples = []
p_samples = []
for _ = 1:1
    PClean.run_smc!(trace, :Obs, row_id, PClean.InferenceConfig(40, 5))
    r_ = copy(trace.tables[:Obs].rows[row_id])
    println(extractor(r_))
    # info = extractor(r_)
    # if info[1] in existing_observations
    #   push!(samples, extractor(r_))
    # end
    # if info[1][1] in existing_physicians
    #   push!(p_samples, etractor(r_))
    # end
end

histograms(samples)
build_response(samples)[2]

johns = find_person(trace, firstname = "JOHN")
john_id = rand(keys(johns))
physician_name(johns[john_id])
john_businesses = extractor.(values(find_business(trace, john_id)))
