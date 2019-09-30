__precompile__()
module PClean

using Gen

# Special distributions
include("distributions/add_typos.jl")
include("distributions/choose_proportionally.jl")
include("parameters/params.jl")

include("dsl/syntax.jl")

include("cleaning/clean.jl")
include("cleaning/importance.jl")
include("cleaning/smc.jl")

end # module
