abstract type HamiltonianModel end

struct Ising <: HamiltonianModel 
    Ni::Int
    Nj::Int
end
Ising() = Ising(1,1)

mutable struct Ising22 <: HamiltonianModel
    Ni::Int
    Nj::Int 
    r::Real
    n::Real
end
Ising22(r) = Ising22(2,2,r,1)

struct Ising33 <: HamiltonianModel 
    Ni::Int
    Nj::Int
    r::Real
end
Ising33(r) = Ising33(3,3,r)