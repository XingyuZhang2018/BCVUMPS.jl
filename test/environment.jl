using BCVUMPS
using BCVUMPS:qrpos,lqpos,leftorth,rightorth,leftenv,FLmap,rightenv,FRmap,ACenv,ACmap,Cenv,Cmap,LRtoC,ALCtoAC,ACCtoALAR,error,obs2x2FL,obs2x2FR,bigleftenv,BgFLmap,bigrightenv,BgFRmap,FLmapK,FRmapK
using CUDA
using LinearAlgebra
using Random
using Test
using OMEinsum
CUDA.allowscalar(false)

@testset "qr with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    A = atype(rand(dtype, 4,4))
    Q, R = qrpos(A)
    @test Array(Q*R) ≈ Array(A)
    @test all(real.(diag(R)) .> 0)
    @test all(imag.(diag(R)) .≈ 0)
end

@testset "lq with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    Random.seed!(100)
    A = atype(rand(dtype, 4,4))
    L, Q = lqpos(A)
    @test Array(L*Q) ≈ Array(A)
    @test all(real.(diag(L)) .> 0)
    @test all(imag.(diag(L)) .≈ 0)
end

@testset "leftorth and rightorth with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 5, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
    end
    AL, L, λ = leftorth(A)
    R, AR, λ = rightorth(A)

    for j = 1:Nj,i = 1:Ni
        M = ein"cda,cdb -> ab"(AL[i,j],conj(AL[i,j]))
        @test (Array(M) ≈ I(D))

        LA = reshape(L[i,j] * reshape(A[i,j], D, d*D), d*D, D)
        ALL = reshape(AL[i,j], d*D, D) * L[i,j] * λ[i,j]
        @test (Array(ALL) ≈ Array(LA))

        M = ein"acd,bcd -> ab"(AR[i,j],conj(AR[i,j]))
        @test (Array(M) ≈ I(D))

        AxR = reshape(reshape(A[i,j], d*D, D)*R[i,j], D, d*D)
        RAR = R[i,j] * reshape(AR[i,j], D, d*D) * λ[i,j]
        @test (Array(RAR) ≈ Array(AxR))
    end
end

@testset "leftenv and rightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64], Ni = [2], Nj = [2]
    Random.seed!(50)
    D, d = 5, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    M = Array{atype{dtype,4},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
        M[i,j] = atype(rand(dtype, d, d, d, d))
    end

    AL, = leftorth(A)
    λL,FL = leftenv(AL, M)
    _, AR, = rightorth(A)
    λR,FR = rightenv(AR, M)

    for j = 1:Nj, i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        @test λL[i,j] * FL[i,j] ≈ FLmap(AL[i,:], AL[ir,:], M[i,:], FL[i,j], j)
        @test λR[i,j] * FR[i,j] ≈ FRmap(AR[i,:], AR[ir,:], M[i,:], FR[i,j], j)
    end
end

@testset "ACenv and Cenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64], Ni = [2], Nj = [2]
    Random.seed!(50)
    D, d = 5, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    M = Array{atype{dtype,4},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
        M[i,j] = atype(rand(dtype, d, d, d, d))
    end

    AL, L = leftorth(A)
    λL,FL = leftenv(AL, M)
    R, AR, = rightorth(A)
    λR,FR = rightenv(AR, M)

    C = LRtoC(L, R)
    AC = ALCtoAC(AL, C)

    λAC, AC = ACenv(AC, FL, M, FR)
    λC, C = Cenv(C, FL, FR)
    for j = 1:Nj, i = 1:Ni
        jr = j + 1 - Nj * (j==Nj)
        @test λAC[i,j] * AC[i,j] ≈ ACmap(AC[i,j], FL[:,j], FR[:,j], M[:,j], i)
        @test λC[i,j] * C[i,j] ≈ Cmap(C[i,j], FL[:,jr], FR[:,j], i)
    end
end

@testset "bcvumps unit test with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64], Ni = [2], Nj = [2]
    Random.seed!(50)
    D, d = 5, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    M = Array{atype{dtype,4},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
        M[i,j] = atype(rand(dtype, d, d, d, d))
    end

    AL, L = leftorth(A)
    λL,FL = leftenv(AL, M)
    R, AR, = rightorth(A)
    λR,FR = rightenv(AR, M)

    C = LRtoC(L,R)

    AL, C, AR = ACCtoALAR(AL, C, AR, M, FL, FR)
    err = error(AL,C,FL,M,FR)
    @test err !== nothing
end

@testset "obsleftenv and obsrightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64], Ni = [2], Nj = [2]
    Random.seed!(50)
    D, d = 5, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    M = Array{atype{dtype,4},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
        M[i,j] = atype(rand(dtype, d, d, d, d))
    end

    AL, = leftorth(A)
    λL,FL = obs2x2FL(AL, M)
    _, AR, = rightorth(A)
    λR,FR = obs2x2FR(AR, M)

    for j = 1:Nj, i = 1:Ni
        ir = Ni + 1 - i
        @test λL[i,j] * FL[i,j] ≈ FLmapK(AL[i,:], AL[ir,:], M[i,:], FL[i,j], j)
        @test λR[i,j] * FR[i,j] ≈ FRmapK(AR[i,:], AR[ir,:], M[i,:], FR[i,j], j)
    end
end

@testset "bigleftenv and bigrightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64], Ni = [2], Nj = [2]
    Random.seed!(50)
    D, d = 5, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    M = Array{atype{dtype,4},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
        M[i,j] = atype(rand(dtype, d, d, d, d))
    end

    AL, = leftorth(A)
    λL,BgFL = bigleftenv(AL, M)
    _, AR, = rightorth(A)
    λR,BgFR = bigrightenv(AR, M)

    for j = 1:Nj, i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        irr = i + 2 - Ni * (i + 2 > Ni)
        @test λL[i,j] * BgFL[i,j] ≈ BgFLmap(AL[i,:], AL[i,:], M[i,:], M[ir,:], BgFL[i,j], j)
        @test λR[i,j] * BgFR[i,j] ≈ BgFRmap(AR[i,:], AR[i,:], M[i,:], M[ir,:], BgFR[i,j], j)
    end
end