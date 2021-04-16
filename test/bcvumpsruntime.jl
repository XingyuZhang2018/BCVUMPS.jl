using Test
using BCVUMPS
using Random

@testset "$(Ni)x$(Nj) bcvumps" for Ni = [1,2,3], Nj = [1,2,3]
    @test SquareLattice <: AbstractLattice

    M = Array{Array,2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        M[i,j] = rand(2,2,2,2)
    end
    rt = SquareBCVUMPSRuntime(M, Val(:random), 2)
    env = bcvumps(rt; tol=1e-10, maxiter=100)
    @test env !== nothing
end




