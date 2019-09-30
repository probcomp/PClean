using Gen
using StringDistances

# Custom distribution for add_typos
struct AddTypos <: Gen.Distribution{String} end

"""
    add_typos(s::String)
Samples a string near `s` (in Levenshtein edit distance).
"""
const add_typos = AddTypos()
export add_typos

function Gen.logpdf(::AddTypos, s::String, obs::String)
    approx_num_possible_typos = length(s)*3
    -log(approx_num_possible_typos) * Float64(StringDistances.evaluate(StringDistances.DamerauLevenshtein(), s, obs))
end

function Gen.random(::AddTypos, s::String)
    # For now, to sample, just return the string
    s
end

(::AddTypos)(x) = random(AddTypos(), x)

Gen.has_output_grad(::AddTypos) = false
Gen.has_argument_grads(::AddTypos) = (false,)
