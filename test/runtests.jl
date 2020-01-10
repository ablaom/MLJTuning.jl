using Distributed
addprocs(2)

@everywhere begin
using MLJTuning
using MLJBase
using Test
using Random
end

@testset "utilities" begin
  @test include("utilities.jl")
end

