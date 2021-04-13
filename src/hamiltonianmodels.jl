abstract type HamiltonianModel end

@doc raw"
    hamiltonian(model<:HamiltonianModel)

return the hamiltonian of the `model` as a two-site tensor operator.
"
function hamiltonian end

struct Ising <: HamiltonianModel 
    Ni::Int
    Nj::Int
end
Ising() = Ising(1,1)

struct Ising22 <: HamiltonianModel
    Ni::Int
    Nj::Int 
    r::Real
end
Ising22(r) = Ising22(2,2,r)

struct Ising33 <: HamiltonianModel 
    Ni::Int
    Nj::Int
    r::Real
end
Ising33(r) = Ising33(3,3,r)