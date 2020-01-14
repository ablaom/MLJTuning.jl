using Distributed
addprocs(2)

@everywhere begin
using Test
using MLJTuning
using MLJBase
end

include("test_utilities.jl")

print("Loading some models for testing...")
# load `Models` module containing models implementations for testing:
@everywhere include("models.jl")
print("\r                                           \r")

@testset "utilities" begin
    @test include("utilities.jl")
end

@testset "tuned_models.jl" begin
    @test include("tuned_models.jl")
end
