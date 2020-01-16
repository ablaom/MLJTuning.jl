const CartesianRange = AbstractVector{<:ParamRange}
const ResolutionVector = AbstractVector{<:Union{Nothing,Integer}}

MLJBase.iterator(r::NominalRange, ::Nothing) = iterator(r)


"""
    MLJTuning.grid(prototype, ranges, resolutions [, rng])

Given an iterable `ranges` of `ParamRange` objects, and an iterable
`resolutions` of the same length, return a vector of models generated
by cloning and mutating the hyperparameters (fields) of `prototype`,
according to the Cartesian grid defined by the specifed
one-dimensional `ranges` (`ParamRange` objects) and specified
`resolutions`. A resolution of `nothing` for a `NominalRange`
indicates that all values should be used.

Specification of an `AbstractRNG` object `rng` implies shuffling of
the results. Otherwise models are ordered, with the first
hyperparameter referenced cycling fastest.

"""
grid(prototype::Model, ranges, resolutions, rng::AbstractRNG) =
    shuffle(rng, grid(prototype, ranges, resolutions))

function grid(prototype::Model, ranges, resolutions)

    iterators = broadcast(iterator, ranges, resolutions)

    A = MLJBase.unwind(iterators...)

    N = size(A, 1)
    map(1:N) do i
        clone = deepcopy(prototype)
        for k in eachindex(ranges)
            field = ranges[k].field
            recursive_setproperty!(clone, field, A[i,k])
        end
        clone
    end
end

