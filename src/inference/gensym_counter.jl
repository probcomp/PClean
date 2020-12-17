mutable struct GensymCounter
    i::Int
end

const gensym_counter = GensymCounter(0)

function reset_gensym_counter!(i)
    gensym_counter.i = i
end

function pclean_gensym!(base::String="row")
    gensym_counter.i += 1
    Symbol("$(base)_$(gensym_counter.i)")
end
