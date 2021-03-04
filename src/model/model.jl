const ClassID = Symbol

"""
    PCleanNode

An abstract type representing nodes in a PCleanClass's dependency graph.
"""
abstract type PCleanNode end

"""
    VertexID

An integer index to a vertex in a PCleanClass's DAG.
"""
const VertexID = Int

"""
    AbsoluteVertexID

An absolute vertex ID unambiguously identifies a DAG node within a PClean 
model, by specifying a class ID and a vertex ID within the class.
"""
struct AbsoluteVertexID
    class   :: ClassID
    node_id :: VertexID
end

"""
    Path

Represents a chain of reference slots: the class path[i].class refers to 
class path[i-1].class via its reference slot path[i].node_id. The ultimate
target of the path is implicitly determined by path[1].node_id's target in 
the path[1].class class.
"""
const Path = Vector{AbsoluteVertexID}



"""
    InjectiveVertexMap

When class A is referenced by class B, class B's DAG includes
nodes for each of class A's nodes. An InjectiveVertexMap maps
node IDs in class A to their corresponding node IDs in class B.
A node of type T in class A will be a node of type 
SubmodelNode{T} in class B, or, if the slot chain connecting
B to A is longer than one hop, SubmodelNode{...SubmodelNode{T}...}.
"""
const InjectiveVertexMap = Dict{VertexID, VertexID}


mutable struct PitmanYorParams
    strength :: Float64
    discount :: Float64
end

"""
    Step{T}

A node in a `Plan` tree. The `rest` field stores the tree's children
(the "rest" of the plan), and is intended to be of type `Plan`. The type
parameter T is a workaround for Julia's poor support for defining mutually
recursive types.
"""
struct Step{T}
    idx :: VertexID
    rest :: T
end

"""
    Plan

A `Plan` is a forest of trees with integer-valued nodes.
Together, these nodes cover all VertexIDs of a particular subproblem
within a PCleanClass. Any two nodes are conditionally independent given
their common ancestors.
"""
struct Plan
    steps :: Vector{Step{Plan}}
end

"""
    PCleanClass

"""
struct PCleanClass
    # Dependency graph.
    graph :: DiGraph
  
    # Maps vertex numbers to nodes
    nodes :: Vector{PCleanNode}

    # Vertex ID(s) of the field(s) on which to index records of this class
    # for fast lookup, if any. Indices require more memory, but can speed
    # up inference if it is often the fact that a certain field is trusted
    # to be clean, and observed.
    hash_keys :: Vector{VertexID}
  
    # Partitions a subset of the graph nodes into "blocks", corresponding
    # to subproblems that SMC will solve sequentially. Static nodes, i.e.
    # those corresponding to parameters or references to other classes,
    # are not in any block.
    blocks :: Vector{Vector{VertexID}}

    # For each block, a corresponding enumeration plan.
    plans  :: Vector{Plan}

    # For each block, a dictionary mapping a "missingness pattern"
    # (set of observed vertexIDs) to a compiled proposal function.
    # The dictinaries begin empty and are filled just-in-time during inference.
    compiled_proposals :: Vector{Dict{Set{VertexID}, Function}}
  
    # Maps symbol names in the user's class declaration to IDs of the nodes that compute them.
    # This is not just debugging information: these names are the mechanism by which queries and
    # other classes refer to an object's properties and reference slots.
    names :: Dict{Symbol, VertexID}
  
    # Each incoming_reference corresponds to a particular path starting from
    # some other PClean class A and ending at this one. The InjectiveVertexMap
    # maps this class's vertex IDs to the corresponding SubmodelNode IDs in the
    # (perhaps indirectly) referring class A.
    incoming_references :: Dict{Path, InjectiveVertexMap}
  
    # Pitman-Yor Parameters
    initial_pitman_yor_params :: PitmanYorParams
end
  
  

##############
# NODE TYPES #
##############

# Represents a deterministic computation.
struct JuliaNode <: PCleanNode
    f :: Function
    arg_node_ids :: Vector{VertexID}
end

# Represents a random choice from a primitive distribution.
struct RandomChoiceNode <: PCleanNode
    dist :: PCleanDistribution
    arg_node_ids :: Vector{VertexID}
end

# Represents a learned parameter's declaration.
struct ParameterNode <: PCleanNode
    make_parameter :: Function
end

# Represents a node that selects a row at random from
# another table. Points to that other table via learned_table_param.
struct ForeignKeyNode <: PCleanNode
    target_class :: ClassID
    # maps node ids in the target class to node IDs in the current class,
    # and is used to initialize parameter values.
    vmap :: InjectiveVertexMap
end

struct SubmodelNode <: PCleanNode
    foreign_key_node_id :: VertexID # can be used to lookup the gensym
    subnode_id          :: VertexID # the id of this node in the other class; used to look up values in trace
    subnode             :: PCleanNode # has args that are set according to *this* model's indices
end

# Represents a computation that uses the values in this model,
# but are not technically a part of this model.
struct ExternalLikelihoodNode <: PCleanNode
    path :: Path
    # ID of this node in the referring class.
    external_node_id :: VertexID

    # an ExternalLikelihood node should *only* be a JuliaNode
    # or a random choice node (or a foreign key node, though
    # that feature — DPMem-style invocation of a class — is not yet implemented.)
    # Unlike a SubmodelNode's `subnode`, an ExternalLikelihoodNode's `external_node` may reference
    # VertexIDs *not* valid for the current class, but rather the referring class.
    external_node :: Union{JuliaNode, RandomChoiceNode, ForeignKeyNode}
end


# The model itself needn't store the dependency structure,
# I think...
struct PCleanModel
    classes :: Dict{ClassID, PCleanClass}
    class_order :: Vector{ClassID}
end

