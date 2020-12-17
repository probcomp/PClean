module PClean

using Distributions
using LightGraphs
using CSV

include("utils.jl")

# Distributions
include("distributions/distributions.jl")

# Models
include("tables/table_model.jl")
include("tables/table_trace.jl")
include("tables/dependency_tracking.jl")

# DSL
include("dsl/builder.jl")
include("dsl/syntax.jl")
include("dsl/query.jl")

# Inference
include("inference/gensym_counter.jl")
include("inference/infer_config.jl")
include("inference/proposal_row_state.jl")
include("inference/block_proposal.jl")
include("inference/row_inference.jl")
include("inference/table_inference.jl")
include("inference/compiler.jl")
# include("inference/instrumented_inference.jl")

# Analysis
include("analysis.jl")

end # module
