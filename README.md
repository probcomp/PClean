# PClean

[![Build Status](https://travis-ci.com/probcomp/PClean.svg?branch=master)](https://travis-ci.com/probcomp/PClean)

PClean: A Domain-Specific Probabilistic Programming Language for Bayesian Data Cleaning

*Warning: This is a rapidly evolving research prototype.*

PClean was created at the [MIT Probabilistic Computing Project](http://probcomp.csail.mit.edu/).

If you use PClean in your research, please cite the our 2021 AISTATS paper:

PClean: Bayesian Data Cleaning at Scale with Domain-Specific Probabilistic Programming. Lew, A. K.; Agrawal, M.; Sontag, D.; and Mansinghka, V. K. (2021, March).
In International Conference on Artificial Intelligence and Statistics (pp. 1927-1935). PMLR. ([pdf](http://proceedings.mlr.press/v130/lew21a/lew21a.pdf))

## Using PClean


To use PClean, create a Julia file with the following structure:

```julia
using PClean
using DataFrames: DataFrame
import CSV

# Load data
data = CSV.File(filepath) |> DataFrame

# Define PClean model
PClean.@model MyModel begin
    @class ClassName1 begin
        ...
    end

    ...
    
    @class ClassNameN begin
        ...
    end
end

# Align column names of CSV with variables in the model.
# Format is ColumnName CleanVariable DirtyVariable, or, if
# there is no corruption for a certain variable, one can omit
# the DirtyVariable.
query = @query MyModel.ClassNameN [
  HospitalName hosp.name             observed_hosp_name
  Condition    metric.condition.desc observed_condition
  ...
]

# Configure observed dataset
observations = [ObservedDataset(query, data)]

# Configuration
config = PClean.InferenceConfig(1, 2; use_mh_instead_of_pg=true)

# SMC initialization
state = initialize_trace(observations, config)

# Rejuvenation sweeps
run_inference!(state, config)

# Evaluate accuracy, if ground truth is available
ground_truth = CSV.File(filepath) |> CSV.DataFrame
results = evaluate_accuracy(data, ground_truth, state, query)

# Can print results.f1, results.precision, results.accuracy, etc.
println(results)

# Even without ground truth, can save the entire latent database to CSV files:
PClean.save_results(dir, dataset_name, state, observations)
```

Then, from this directory, run the Julia file.

```
JULIA_PROJECT=. julia my_file.jl
```

To learn to write a PClean model, see [our paper](http://proceedings.mlr.press/v130/lew21a/lew21a.pdf), but note
the surface syntax changes described below.

## Differences from the paper

As a DSL embedded into Julia, our implementation of the PClean language has some differences, in terms of surface syntax,
from the stand-alone syntax presented in our paper:

(1) Instead of `latent class C ... end`, we write `@class C begin ... end`.

(2) Instead of `subproblem begin ... end`, inference hints are given using ordinary
    Julia `begin ... end` blocks.

(3) Instead of `parameter x ~ d(...)`, we use `@learned x :: D{...}`. The set of
    distributions D for parameters is somewhat restricted.

(4) Instead of `x ~ d(...) preferring E`, we write `x ~ d(..., E)`.

(5) Instead of `observe x as y, ... from C`, write `@query ModelName.C [x y; ...]`.
    Clauses of the form `x z y` are also allowed, and tell PClean that the model variable
    `C.z` represents a clean version of `x`, whose observed (dirty) version is modeled
    as `C.y`. This is used when automatically reconstructing a clean, flat dataset.

The names of built-in distributions may also be different, e.g. `AddTypos` instead of `typos`,
and `ProportionsParameter` instead of `dirichlet`.