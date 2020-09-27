using BCVUMPS
using Test, LinearAlgebra, TensorOperations, KrylovKit, Random

include("3x3mpo.jl")

@testset "3x3M,Me" begin
    include("3x3M,ME.jl")
end

@testset "orthogonality" begin
    include("orth.jl")
end

@testset "environment" begin
    include("env.jl")
end

@testset "vumpsstep" begin
    include("vumpsstep.jl")
end

