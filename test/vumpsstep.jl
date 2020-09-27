using Test

using LinearAlgebra, TensorOperations, KrylovKit, Random

@testset "vumpsstep" for Ni = [2,3],Nj = [2,3]
    β = 0.1
    M, ME_row, ME_col,λM,λME_row,λME_col= classicalisingmpo(β; r = 1.0)
    D = 50
    A = Array{Array,2}(undef, Ni, Nj)
    for i = 1:Ni,j = 1:Nj
        Random.seed!(1234)
        A[i,j] = randn(D, 2, D) + im*randn(D, 2, D)
    end
    AL, L ,λL = BCVUMPS.leftorth(A)
    R, AR,λR = BCVUMPS.rightorth(A,L)
    FL3,FR3,λFL3,λFR3 = BCVUMPS.env3!(AL,AR, M)
    FL4,FR4,λFL4,λFR4 = BCVUMPS.env4!(AL,AR, M)
    C = Array{Array,2}(undef, Ni,Nj)
    for i = 1:Ni,j = 1:Nj
        jr = j + 1 - Nj * (j+1>Nj)
        C[i,j] = L[i,j] * R[i,jr]
    end
    λ, AL, C, AR, errL, errR = BCVUMPS.vumpsstep!(AL,C,FL3,FR3,M)
    for j = 1:Nj,i = 1:Ni
        @tensor AL2[b, c] := AL[i,j][a,s,b]*conj(AL[i,j][a,s,c])
        @test Array(AL2) ≈ I(D)
        @tensor AR2[b, c] := AR[i,j][b,s,a]*conj(AR[i,j][c,s,a])
        @test Array(AR2) ≈ I(D)
    end
end