using BCVUMPS
using BCVUMPS: model_tensor
using Test
using OMEinsum

@testset "exampletensor" begin
    β = rand()
    M = model_tensor(Ising(),β)
    @test size(M) == (1,1)
    for Nj = 1:3, Ni = 1:3
        M = model_tensor(Ising(Ni,Nj),β)
        @test size(M) == (Ni,Nj)
        @test M[1,1] ≈ M[Ni,Nj]
    end

    @test Ising22(1).Ni == 2
    @test Ising22(1).Nj == 2
    M = model_tensor(Ising22(1),β)
    @test M[1,1] ≈ M[1,2] ≈ M[2,1] ≈ M[2,2]

    M = model_tensor(Ising22(2),β)
    @test M[1,2] ≈ permutedims(M[1,1],[3,2,1,4])
    @test M[2,1] ≈ permutedims(M[1,1],[1,4,3,2])
    @test M[2,2] ≈ permutedims(M[1,1],[3,4,1,2])
end
