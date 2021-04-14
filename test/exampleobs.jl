using Test
using BCVUMPS
using BCVUMPS:bcvumps_env,magnetisation,magofβ,energy,eneofβ,Z,Zofβ
using Random

@testset "$(Ni)x$(Nj) ising" for Ni = [1,2,3], Nj = [1,2,3]
    Random.seed!(100)
    model = Ising(Ni,Nj)
    for β = 0.2:0.2:0.8
        env = bcvumps_env(model, β, 2; tol=1e-10, maxiter=20, verbose = false)
        @test isapprox(magnetisation(env,model,β), magofβ(model,β), atol=1e-5)
        @test isapprox(energy(env,model,β), eneofβ(model,β), atol=1e-2)
        @test isapprox(Z(env), Zofβ(model,β), atol=1e-3)
    end
end

@testset "J1-J2-2x2-Ising" begin
    Random.seed!(100)
    model = Ising22(1)
    for β = 0.2:0.2:0.8
        env = bcvumps_env(model, β, 2; tol=1e-10, maxiter=20, verbose = false)
        @test isapprox(magnetisation(env, model, β), magofβ(Ising(),β), atol=1e-5)
        @test isapprox(energy(env, model, β), eneofβ(Ising(),β), atol=1e-2)
        @test isapprox(Z(env), Zofβ(Ising(),β), atol=1e-3)
    end

    Random.seed!(100)
    model = Ising22(2)
    for β = 0.4:0.2:0.8
        env = bcvumps_env(model, β, 10; tol=1e-10, maxiter=20, verbose = false)
        mag22,ene22 = MCMC(model,16,β,10000,100000)
        @test isapprox(magnetisation(env,model,β), mag22, atol=1e-3)
        @test isapprox(energy(env,model,β), ene22, atol=1e-3)
    end
end