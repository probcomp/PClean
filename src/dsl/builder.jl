using LightGraphs


############
# Blocking #
############

@enum BlockStatus begin
    open   # all new nodes added to most recent block
    closed # current block is over; next statement goes in new block
end

function begin_block!(b, class)
    push!(b.model.classes[class].blocks, [])
    b.block_status = open
end

function end_block!(b)
    b.block_status = closed
end



"""
    PCleanModelBuilder

A PCleanModelBuilder is a partially constructed PClean model.
"""
mutable struct PCleanModelBuilder
    model::PCleanModel
    block_status::BlockStatus
end

####################
#  Start New Class #
####################
function add_new_class!(builder::PCleanModelBuilder, class::ClassID)
    builder.model.classes[class] = PCleanClass(DiGraph(), [], [], [], [], [], 
                                                Dict(), Dict(), PitmanYorParams(1.0, 0.0))
    push!(builder.model.class_order, class)
end

############################
# Name resolution routines #
############################

# Split an expression like x.y.z.w into a head x and a tail y.z.w.
# [NOTE: Julia naturally interprets as (x.y.z).w, necessitating the somewhat
# roundabout code below. We could rewrite resolve_dot_expression to work with
# the existing recursive structure. But performance of this parsing code is not
# critical.]
function split_first_dot(expr)
    first_arg  = expr.args[1] isa QuoteNode ? expr.args[1].value : expr.args[1]
    second_arg = expr.args[2] isa QuoteNode ? expr.args[2].value : expr.args[2]
    if expr.head == :. && first_arg isa Symbol
        return first_arg, second_arg
    end
    x, y = split_first_dot(first_arg)
    return x, Expr(:., y, second_arg)
end

# Resolve a dot expression (x.y.z.w...) in the context of a particular class.
function resolve_dot_expression(model::PCleanModel, class::ClassID, argument)
    class_model = model.classes[class]

    # Base case: no reference slot chain.
    if argument isa Symbol
        return class_model.names[argument]
    end

    # Recursive case: follow a reference slot.
    reference_slot_name, accessor = split_first_dot(argument)
    reference_slot_node = class_model.nodes[class_model.names[reference_slot_name]]
    target_class = reference_slot_node.target_class
    
    return reference_slot_node.vmap[resolve_dot_expression(model, target_class, accessor)]
end

# Process an argument to a distribution. An argument is either:
#    * a symbol (directly resolved to a local variable in the same class)
#    * an Expr  (must be a dot expression of the form x.y.z...)
#    * a Tuple{Vector{Argument}, Function}, whose first element is a list of arguments (as defined in this comment)
#      and whose second element is a function mapping values of those arguments to the value of this argument.
# `resolve_argument!` adds any necessary Julia nodes to the class's graph and 
# returns a node ID in the class that computes the argument's value.
function resolve_argument!(b::PCleanModelBuilder, class::ClassID, argument)
    class_model = b.model.classes[class]
    if argument isa Symbol
        @assert haskey(class_model.names, argument)
        return class_model.names[argument]
    elseif argument isa Expr
        @assert argument.head == :.
        return resolve_dot_expression(b.model, class, argument)
    else
        syms, f = argument
        add_julia_node!(b, class, gensym(repr(argument)), syms, f)
        return nv(class_model.graph)
    end
end

#####################
# Guaranteed Values #
#####################
function add_guaranteed!(b::PCleanModelBuilder, class::ClassID, name::Union{Symbol,Expr})
    push!(b.model.classes[class].hash_keys, resolve_argument!(b, class, name))
end

#####################
# Foreign Key Nodes #
#####################

# The `copy_node` functions create versions of a node where all references to 
# other nodes are shifted by an amount `v`. These are used when copying the nodes
# from the target class of a reference slot into the source class as SubmodelNodes.
copy_node(n::JuliaNode, v::Int) = JuliaNode(n.f, map(x -> x + v, n.arg_node_ids))
copy_node(n::RandomChoiceNode, v::Int) = RandomChoiceNode(n.dist, map(x -> x + v, n.arg_node_ids))
copy_node(n::ParameterNode, v::Int) = n
copy_node(n::ForeignKeyNode, v::Int) = ForeignKeyNode(n.target_class,
                                                      InjectiveVertexMap(i => j + v for (i, j) in n.vmap))
copy_node(n::SubmodelNode, v::Int) = SubmodelNode(n.foreign_key_node_id + v, n.subnode_id, copy_node(n.subnode, v))


function add_foreign_key!(b::PCleanModelBuilder, source_class::ClassID, name::Symbol, target_class::Symbol)
    source_model = b.model.classes[source_class]
    target_model = b.model.classes[target_class]

    # First, add the foreign key itself.
    add_vertex!(source_model.graph)
    v = nv(source_model.graph)
    source_model.names[name] = v
    target_model_nodes = filter(node -> !(node isa ExternalLikelihoodNode), 
                                target_model.nodes)
    push!(source_model.nodes,
           ForeignKeyNode(target_class, 
                          Dict(i => i + v for i = 1:length(target_model_nodes))))


    # The foreign key has an incoming edge from any *other* foreign key targeting the same class...
    parent_nodes = [(i, n) for (i, n) in enumerate(source_model.nodes) if n isa ForeignKeyNode && n.target_class == target_class]
    # It is also the case that my submodel nodes *could* depend on its submodel nodes.
    # Capturing this dependency in a fine-grained manner might allow more interesting blocking 
    # schemes that cut across classes, but for now, we'll capture it by saying that every submodel node
    # for the same class causes our reference slot node.
    for (i, parent_node) in parent_nodes
        add_edge!(source_model.graph, i, v)
        for submodel_node in values(parent_node.vmap)
            add_edge!(source_model.graph, submodel_node, v)
        end
    end
    
    # Then, add nodes from original graph.
    for (i, node) in enumerate(target_model_nodes)
        add_vertex!(source_model.graph)
        push!(source_model.nodes, SubmodelNode(v, i, copy_node(node, v)))
        add_edge!(source_model.graph, v, i + v)
    end
  
    # Add edges from original graph.
    limit = nv(source_model.graph)
    for e in edges(target_model.graph)
        if e.src + v <= limit && e.dst + v <= limit
            add_edge!(source_model.graph, e.src + v, e.dst + v)
        end
    end
  
    # Add newly created nodes to blocks.
    all_sampled_nodes = vcat([v], [filter(x -> x <= limit, map(x -> x + v, block)) for block in target_model.blocks]...)

    if b.block_status == open
        push!(source_model.blocks[end], all_sampled_nodes...)
    elseif b.block_status == closed
        push!(source_model.blocks, all_sampled_nodes)
        b.block_status = open
    end
end


##############
# Parameters #
##############

function add_basic_parameter!(b::PCleanModelBuilder, class::ClassID, name::Symbol, t::Type{T}, args...) where T <: BasicParameter
    # Add a node to the graph, and give it the right name.
    class_model = b.model.classes[class]
    add_vertex!(class_model.graph)
    v = nv(class_model.graph)
    class_model.names[name] = v
  
    # Construct the ParameterNode. This requires a make_parameter function.
    # which requires a prior, on which we can call initialize_parameter.
    parameter_prior = length(args) == 1 && args[1] isa ParameterPrior ? args[1] : default_prior(t, args...)
    push!(class_model.nodes, ParameterNode(() -> initialize_parameter(t, parameter_prior)))
end

function add_indexed_parameter!(b::PCleanModelBuilder, class::ClassID, name::Symbol, t::Type{T}, args...) where T <: Parameter
    class_model = b.model.classes[class]
    add_vertex!(class_model.graph)
    v = nv(class_model.graph)
    class_model.names[name] = v
    prior = length(args) == 1 && args[1] isa ParameterPrior ? args[1] : default_prior(t, args...)
    push!(class_model.nodes, ParameterNode(() -> IndexedParameter(prior, Dict{Any,T}())))
end

##########################
# Julia and Choice Nodes #
##########################

function add_julia_node!(b::PCleanModelBuilder, class::ClassID, name::Symbol, arguments::Vector, f::Function)
    class_model = b.model.classes[class]
    
    # It happens to be the case that this will only
    # involve symbols and dot expressions; the parser handles the rest.
    arg_indices = [resolve_argument!(b, class, arg) for arg in arguments]
    
    # Add the node
    add_vertex!(class_model.graph)
    v = nv(class_model.graph)
    class_model.names[name] = v
    for arg in arg_indices
        add_edge!(class_model.graph, arg, v)
    end
    push!(class_model.nodes, JuliaNode(f, arg_indices))
  
    # Place it in a block.
    if b.block_status == closed
        push!(class_model.blocks, [v])
        b.block_status = open
    else
        push!(class_model.blocks[end], v)
    end
end


function add_choice_node!(b::PCleanModelBuilder, class::ClassID, name::Symbol, dist::PCleanDistribution, arguments::Vector)
    class_model = b.model.classes[class]
    
    # Resolve arguments
    arg_indices = [resolve_argument!(b, class, a) for a in arguments]
  
    # Add vertex and record name
    add_vertex!(class_model.graph)
    v = nv(class_model.graph)
    class_model.names[name] = v
  
    # Create node
    for arg in arg_indices
        add_edge!(class_model.graph, arg, v)
    end
    push!(class_model.nodes, RandomChoiceNode(dist, arg_indices))
  
    # Place it in a block.
    if b.block_status == open
        push!(class_model.blocks[end], v)
    elseif b.block_status == closed
        push!(class_model.blocks, [v])
        b.block_status = open
    end
end

#######################
#    External Nodes   #
#######################

function add_external_nodes!(model_node, node_id, block_id, path, target_model, source_model, added_node_ids, from=nothing)
    @assert model_node isa ParameterNode || model_node isa SubmodelNode
end


function add_external_nodes!(model_node::Union{JuliaNode,RandomChoiceNode,ForeignKeyNode},
                         node_id, block_id, path, target_model, source_model,
                         added_node_ids, from = nothing)
    
    # If this node has already been processed
    if haskey(added_node_ids, node_id)
        if !isnothing(from)
            add_edge!(target_model.graph, from, added_node_ids[node_id])
        end
        return
    end

    # Add this external_node
    add_vertex!(target_model.graph)
    added_node_ids[node_id] = nv(target_model.graph)
    if !isnothing(from)
        add_edge!(target_model.graph, from, added_node_ids[node_id])
    end
    push!(target_model.blocks[block_id], added_node_ids[node_id])
    push!(target_model.nodes, ExternalLikelihoodNode(path, node_id, model_node))

    # Continue adding, if this is not an absorbing node
    if model_node isa JuliaNode
        for next in outneighbors(source_model.graph, node_id)
            add_external_nodes!(source_model.nodes[next], next, block_id, path, target_model, source_model, added_node_ids, added_node_ids[node_id])
        end
    end
end


# Given a model, a target class, a path ending in that target class,
# and a vmap mapping target class nodes to source class nodes,
# does three things:
#   1. Adds the path and vmap to the target class's incoming_references.
#   2. Adds ExternalLikelihoodNodes to the target class representing the choices of the source class
#   3. Recursively does the same for all paths containing `path` as a prefix.
function process_reference!(model, target_class, path, vmap)

    source_class = path[end].class
    source_model = model.classes[source_class]
    target_model = model.classes[target_class]

    # Add incoming reference
    target_model.incoming_references[path] = vmap

    # Add external_nodes
    # TODO: We *could* store a *list* of paths in each external_node, and only construct
    # one 'external_node set' per class.
    added_node_ids = Dict() # maps source_class nodes to (newly created) target_class nodes
    for (block_idx, block) in reverse(collect(enumerate(target_model.blocks)))
        nodes_from_block = [(i, vmap[i]) for i in block if !(target_model.nodes[i] isa ExternalLikelihoodNode)]

        for (target_node, source_node) in nodes_from_block
            for next in outneighbors(source_model.graph, source_node)
                add_external_nodes!(source_model.nodes[next], next, block_idx, path, target_model, source_model, added_node_ids, target_node)
            end
        end

    end

    # Process paths of length + 1.
    for (v, node) in enumerate(target_model.nodes)
        if node isa ForeignKeyNode
            # Construct new path
            new_path = [AbsoluteVertexID(target_class, v), path...]
            # Construct new vmap
            new_vmap = Dict(i => vmap[j] for (i, j) in node.vmap)
            # Recursively process the new path
            process_reference!(model, node.target_class, new_path, new_vmap)
        end
    end
end

# Calls `process_reference!` for all paths originating at `class`.
function process_references!(model::PCleanModel, class::ClassID)
    for (v, node) in enumerate(model.classes[class].nodes)
        if node isa ForeignKeyNode
            path = [AbsoluteVertexID(class, v)]
            process_reference!(model, node.target_class, path, node.vmap)
        end
    end
end

##################
#     Plans      #
##################

function make_plan(graph, toposort)
    subg, vmap = induced_subgraph(graph, toposort)
    components = connected_components(subg)
    component_toposorts = [filter(v -> in(v, map(i -> vmap[i], c)), toposort) for c in components]
    return Plan(Step{Plan}[Step{Plan}(t[1], make_plan(graph, t[2:end])) for t in component_toposorts])  
end

function make_plans!(model::PCleanModel)

    for (class_id, class_model) in model.classes
        for (i, block) in enumerate(class_model.blocks)
            push!(class_model.plans, make_plan(class_model.graph, block))
            push!(class_model.compiled_proposals, Dict{Set{VertexID}, Function}())
        end
    end

end

##################
#  Finish Model  #
##################

function finish_class!(b::PCleanModelBuilder, class::ClassID)
    process_references!(b.model, class)
    b.block_status = closed
end

function finish_model!(b::PCleanModelBuilder)
    make_plans!(b.model)
    return b.model
end