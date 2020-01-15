const CartesianRange = AbstractVector{<:ParamRange}
const ResolutionVector = AbstractVector{<:Union{Nothing,Integer}}

MLJBase.iterator(r::NominalRange, ::Nothing) = iterator(r)

function models(prototype::Model, cr::CartesianRange,
                resolutions::ResolutionVector)

    iterators = broadcast(iterator, cr, resolutions)

    A = MLJBase.unwind(iterators...)

    N = size(A, 1)
    map(1:N) do i
        clone = deepcopy(prototype)
        for k in eachindex(cr)
            field = cr[k].field
            recursive_setproperty!(clone, field, A[i,k])
        end
        clone
    end
end

