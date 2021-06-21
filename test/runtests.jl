using Base: array_subpadding
using BCVUMPS
using Test

@testset "BCVUMPS.jl" begin
    @testset "fixedpoint" begin
        println("fixedpoint tests running...")
        include("fixedpoint.jl")
    end

    @testset "environment" begin
        println("environment tests running...")
        include("environment.jl")
    end

    @testset "bcvumpsruntime" begin
        println("bcvumpsruntime tests running...")
        include("bcvumpsruntime.jl")
    end

    @testset "example tensors" begin
        println("exampletensors tests running...")
        include("exampletensors.jl")
    end

    @testset "MCMC" begin
        println("MCMC tests running...")
        include("MCMC.jl")
    end

    @testset "example obs" begin
        println("example obs tests running...")
        include("exampleobs.jl")
    end
end

