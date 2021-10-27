using BCVUMPS
using BCVUMPS:bcvumps_env,magnetisation,magofβ,energy,eneofβ,Z,Zofβ,BigZ
using CUDA
using Random
using Test

@testset "$(Ni)x$(Nj) ising with $atype" for atype in [Array, CuArray], Ni = [2], Nj = [2]
    Random.seed!(100)
    model = Ising(Ni,Nj)
    for β = 0.2:0.2:0.8
        @show β
        M = model_tensor(model, β; atype = atype)
        env = bcvumps_env(model, M; atype = atype, χ = 10, miniter = 2, verbose = true)
        @test isapprox(magnetisation(env,model,β), magofβ(model,β), atol=1e-5)
        @test isapprox(energy(env,model,β), eneofβ(model,β), atol=1e-2)
        @test isapprox(Z(env), Zofβ(model,β), atol=1e-3)
    end
end

@testset "$(Ni)x$(Nj) ising up and down with $atype" for atype in [Array], Ni = [2], Nj = [2]
    Random.seed!(100)
    model = Ising(Ni,Nj)
    for β = 0.2
        @show β
        M = model_tensor(model, β; atype = atype)
        env = obs_bcenv(M; χ = 10, maxiter = 10, verbose = true)
        @test isapprox(magnetisation(env,model,β), magofβ(model,β), atol=1e-8)
        @test isapprox(energy(env,model,β), eneofβ(model,β), atol=1e-7)
        # @test isapprox(Z(env), Zofβ(model,β), atol=1e-3)
        # @test isapprox(BigZ(env), Zofβ(model,β), atol=1e-3)
    end
end

@testset "J1-J2-2x2-Ising with $atype{$dtype}" for atype in [Array], dtype in [Float64]
    Random.seed!(100)
    model = Ising22(1)
    for β = 0.2:0.2:0.4
        M = model_tensor(model, β; atype = atype)
        env = obs_bcenv(model, M; atype = atype, χ = 10, miniter = 2, verbose = true)
        @test isapprox(magnetisation(env, model, β), magofβ(Ising(),β), atol=1e-5)
        @test isapprox(energy(env, model, β), eneofβ(Ising(),β), atol=1e-2)
        # @test isapprox(Z(env), Zofβ(Ising(),β), atol=1e-3)
    end

    Random.seed!(100)
    model = Ising22(2)
    for β = 0.4
        M = model_tensor(model, β; atype = atype)
        env = obs_bcenv(model, M; atype = atype, χ = 10, miniter = 10, verbose = true)
        mag22,ene22 = MCMC(model,16,β,10000,100000)
        @test isapprox(magnetisation(env,model,β), mag22, atol=1e-3)
        @test isapprox(energy(env,model,β), ene22, atol=1e-3)
    end
end