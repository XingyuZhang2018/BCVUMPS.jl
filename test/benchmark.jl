using BCVUMPS
using BCVUMPS:bcvumps_env
using BenchmarkTools
using Random
using Test

@testset "$(Ni)x$(Nj) ising" for Ni = [2], Nj = [2]
    Random.seed!(100)
    model = Ising(Ni,Nj)
    @btime bcvumps_env($(model), 0.5, 50; tol=1e-10, maxiter=20, verbose = false)
end