using Gen

struct ChooseProportionally <: Gen.Distribution{Any} end

"""
    choose_proportionally(objects::AbstractArray{T, 1}, probs::AbstractArray{U, 1}) where {U <: Real}
Given a vector of probabilities `probs` where `sum(probs) = 1`, sample an `Int` `i` from the set {1, 2, .., `length(probs)`} with probability `probs[i]`, then return `objects[i]`.
"""
const choose_proportionally = ChooseProportionally()
export choose_proportionally

function Gen.logpdf(::ChooseProportionally, x, objects, probs::AbstractArray{U, 1}) where {U <: Real}
    total = 0
    for i=1:length(objects)
        if objects[i] == x
            total += probs[i]
        end
    end
    return log(total / sum(probs))
end

function Gen.logpdf(::ChooseProportionally, x, objects, probs::Nothing)
    return -log(length(objects))
end

function Gen.logpdf(::ChooseProportionally, x, probs::Dict) where {U <: Real}
    return log(probs[x] / sum(values(probs)))
end


function Gen.logpdf_grad(::ChooseProportionally, x, objects, probs::AbstractArray{U,1})  where {U <: Real}
    grad = zeros(length(probs))
    total = 0
    relevant_indices = []
    for i=1:length(objects)
        if objects[i] == x
            total += probs[i]
            push!(relevant_indices, i)
        end
    end
    for i in relevant_indices
        grad[i] = 1. / total - 1. / sum(probs)
    end
    (nothing, grad)
end

function Gen.random(::ChooseProportionally, objects, probs::AbstractArray{U, 1}) where {U <: Real}
    normalized_probs = probs / sum(values(probs))
    objects[categorical(normalized_probs)]
end

function Gen.random(::ChooseProportionally, objects, probs::Nothing)
    objects[uniform_discrete(1, length(objects))]
end

function Gen.random(::ChooseProportionally, probs::Dict)
    normalized_probs = collect(values(probs)) / sum(values(probs))
    collect(keys(probs))[categorical(normalized_probs)]
end

(::ChooseProportionally)(probs) = random(ChooseProportionally(), probs)

(::ChooseProportionally)(objects, probs) = random(ChooseProportionally(), objects, probs)

Gen.has_output_grad(::ChooseProportionally) = false
Gen.has_argument_grads(::ChooseProportionally) = (false,true)
