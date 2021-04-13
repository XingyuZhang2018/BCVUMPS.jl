using Test
using BCVUMPS
using BCVUMPS:MCMC,magofβ,eneofβ
using Random

@testset "MCMC" begin
    Random.seed!(100)
    for β in [0.6,0.8]
        mag22,ene22 = MCMC(Ising22(1),12,β,10000,100000)
        mag33,ene33 = MCMC(Ising33(1),12,β,10000,100000)
        @test isapprox(mag22, mag33, atol=1e-2)
        @test isapprox(ene22, ene33, atol=1e-2)
        @test isapprox(mag22, magofβ(Ising(),β), atol=1e-2)
        @test isapprox(ene22, eneofβ(Ising(),β), atol=1e-2)
        @test isapprox(mag33, magofβ(Ising(),β), atol=1e-2)
        @test isapprox(ene33, eneofβ(Ising(),β), atol=1e-2)
    end
end

