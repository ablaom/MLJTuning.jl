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

@testset "one_dimensional_ranges" begin
  @test include("one_dimensional_ranges.jl")
end



