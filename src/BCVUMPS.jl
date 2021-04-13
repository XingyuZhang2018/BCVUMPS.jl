module BCVUMPS

using OMEinsum
export hamiltonian, model_tensor, mag_tensor
export Ising, Ising22, Ising33
export bcvumps, bcvumps_env, magnetisation, magofβ, energy, eneofβ
export MCMC

include("hamiltonianmodels.jl")
include("fixedpoint.jl")
include("environment.jl")
include("bcvumpsruntime.jl")
include("exampletensors.jl")
include("MCMC.jl")
include("exampleobs.jl")

end
