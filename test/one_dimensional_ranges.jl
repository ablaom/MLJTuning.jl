import MLJBase: transform, inverse_transform, Deterministic

mutable struct DummyModel <: Deterministic
    K::Int
    metric::Float64
    kernel::Char
end

dummy_model = DummyModel(4, 9.5, 'k')

mutable struct SuperModel <: Deterministic
    lambda::Float64
    model1::DummyModel
    model2::DummyModel
end

dummy1 = DummyModel(1, 9.5, 'k')
dummy2 = DummyModel(2, 9.5, 'k')
super_model = SuperModel(0.5, dummy1, dummy2) 

@testset "params function" begin
    @test params(dummy_model) ==
        (K = 4, metric = 9.5, kernel = 'k')
    params(super_model)

end

@testset "range constructors, scale, iterator" begin
    p1 = range(dummy_model, :K, lower=1, upper=10, scale=:log10) 
    p2 = range(dummy_model, :kernel, values=['c', 'd']) 
    p3 = range(super_model, :lambda, lower=0.1, upper=1, scale=:log2) 
    p4 = range(dummy_model, :K, lower=1, upper=3, scale=x->2x) 
    @test_throws ErrorException range(dummy_model, :K, lower=1, values=['c', 'd'])
    @test_throws ErrorException range(dummy_model, :kernel, upper=10)

    @test scale(p1) == :log10
    @test scale(p2) == :none
    @test scale(p3) == :log2
    @test scale(p4) == :custom
    @test scale(sin) === sin
    @test transform(MLJTuning.Scale, scale(:log), ℯ) == 1
    @test inverse_transform(MLJTuning.Scale, scale(:log), 1) == float(ℯ)

    @test iterator(p1, 5)  == [1, 2, 3, 6, 10]
    @test iterator(p2) == collect(p2.values)
    u = 2^(log2(0.1)/2)
    @test iterator(p3, 3) ≈ [0.1, u, 1]
    @test iterator(p4, 3) == [2, 4, 6]
end

@testset "range constructors for nested parameters" begin
    p1 = range(dummy_model, :K, lower=1, upper=10, scale=:log10) 
    q1 = range(super_model, :(model1.K) , lower=1, upper=10, scale=:log10) 
    @test iterator(q1, 5) == iterator(p1, 5)
    q2 = range
end

true
