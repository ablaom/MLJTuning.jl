module TestRanges

using Test
using MLJBase
using MLJTuning

# `in` for MLJType is overloaded to be `===` based. For purposed of
# testing here, we need `==` based:
function _in(x, itr)::Union{Bool,Missing}
    for y in itr
        ismissing(y) && return missing
        y == x && return true
    end
    return false
end
_issubset(itr1, itr2) = all(_in(x, itr2) for x in itr1)

mutable struct DummyModel <: Deterministic
    lambda::Float64
    metric::Float64
    kernel::Char
end

dummy_model = DummyModel(4, 9.5, 'k')

mutable struct SuperModel <: Deterministic
    K::Int64
    model1::DummyModel
    model2::DummyModel
end

dummy_model = DummyModel(1.2, 9.5, 'k')
super_model = SuperModel(4, dummy_model, deepcopy(dummy_model))

r1 = range(super_model, :(model1.kernel), values=['c', 'd'])
r2 = range(super_model, :K, lower=1, upper=10, scale=:log10)

@testset "models from cartesian range and resolutions" begin

    # with method:
    m1 = MLJTuning.models(super_model, [r1, r2], [nothing, 7])

    # generate all models by hand:
    models1 = [SuperModel(1, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(1, DummyModel(1.2, 9.5, 'd'), dummy_model),
               SuperModel(2, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(2, DummyModel(1.2, 9.5, 'd'), dummy_model),
               SuperModel(3, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(3, DummyModel(1.2, 9.5, 'd'), dummy_model),
               SuperModel(5, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(5, DummyModel(1.2, 9.5, 'd'), dummy_model),
               SuperModel(7, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(7, DummyModel(1.2, 9.5, 'd'), dummy_model),
               SuperModel(10, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(10, DummyModel(1.2, 9.5, 'd'), dummy_model)]

    @test _issubset(models1, m1) && _issubset(m1, models1)

    # with method:
    m2 = MLJTuning.models(super_model, [r1, r2], [1, 7])

    # generate all models by hand:
    models2 = [SuperModel(1, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(2, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(3, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(5, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(7, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(10, DummyModel(1.2, 9.5, 'c'), dummy_model)]

    @test _issubset(models2, m2) && _issubset(m2, models2)

end

end
true
