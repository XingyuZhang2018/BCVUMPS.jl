using LinearAlgebra
using KrylovKit

export AbstractLattice, SquareLattice
abstract type AbstractLattice end
struct SquareLattice <: AbstractLattice end

export BCVUMPSRuntime, SquareBCVUMPSRuntime

# NOTE: should be renamed to more explicit names
"""
    BCVUMPSRuntime{LT}

a struct to hold the tensors during the `bcvumps` algorithm, each is a `Ni` x `Nj` Matrix, containing
- `d × d × d × d'` `M[i,j]` tensor
- `D × d' × D` `AL[i,j]` tensor
- `D × D`     `C[i,j]` tensor
- `D × d' × D` `AR[i,j]` tensor
- `D × d' × D` `FL[i,j]` tensor
- `D × d' × D` `FR[i,j]` tensor
and `LT` is a AbstractLattice to define the lattice type.
"""
struct BCVUMPSRuntime{LT,T,N,AT<:AbstractArray{<:AbstractArray,2},ET}
    M::AT
    AL::ET
    C::ET
    AR::ET
    FL::ET
    FR::ET
    function BCVUMPSRuntime{LT}(M::AT, AL::ET, C::ET, AR::ET,FL::ET, FR::ET) where {LT<:AbstractLattice,AT<:AbstractArray{<:AbstractArray,2},ET<:AbstractArray{<:AbstractArray,2}}
        T, N = eltype(M[1,1]), ndims(M[1,1])
        new{LT,T,N,AT,ET}(M,AL,C,AR,FL,FR)
    end
end

const SquareBCVUMPSRuntime{T,AT} = BCVUMPSRuntime{SquareLattice,T,4,AT}
function SquareBCVUMPSRuntime(M::AT,AL,C,AR,FL,FR) where {AT<:AbstractArray{<:AbstractArray,2}}
    ndims(M[1,1]) == 4 || throw(DimensionMismatch("M dimensions error, should be `4`, got $(ndims(M[1,1]))."))
    BCVUMPSRuntime{SquareLattice}(M,AL,C,AR,FL,FR)
end

@doc raw"
    SquareBCVUMPSRuntime(M::AbstractArray{T,4}, env::Val, χ::Int)

create a `SquareBCVUMPSRuntime` with M-tensor `M`. The `NixNj` `AL,C,AR,FL,FR`
tensors are initialized according to `env`. If `env = Val(:random)`,
the `A[i,j]` is initialized as a random `D×d×D` tensor,and `AL[i,j],C[i,j],AR[i,j]` are the corresponding 
canonical form. `FL,FR` is the left and right environment.

# example

```jldoctest; setup = :(using BCVUMPS)
julia> Ni, Nj = 2, 2;

julia> M = Array{Array,2}(undef, Ni, Nj);

julia> for j = 1:Nj, i = 1:Ni
           M[i,j] = rand(2,2,2,2)
       end

julia> rt = SquareBCVUMPSRuntime(M, Val(:random), 4);

julia> size(rt.AL) == (2,2)
true

julia> size(rt.AL[1,1]) == (4,2,4)
true
```
"
function SquareBCVUMPSRuntime(M::AbstractArray{<:AbstractArray, 2}, env, D::Int; verbose = false)
    return SquareBCVUMPSRuntime(M, _initializect_square(M, env, D; verbose = verbose)...)
end

function _initializect_square(M::AbstractArray{<:AbstractArray, 2}, env::Val{:random}, D::Int; verbose = false)
    T = eltype(M[1,1])
    Ni, Nj = size(M)
    A = Array{Array,2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        d = size(M[i,j],4)
        A[i,j] = rand(T,D,d,D)
    end
    AL,L = leftorth(A)
    R, AR = rightorth(AL)
    _, FL = leftenv!(AL, M)
    _, FR = rightenv!(AR, M)
    C = Array{Array,2}(undef, Ni,Nj)
    for j = 1:Nj,i = 1:Ni
        jr = j + 1 - (j+1>Nj) * Nj
        C[i,j] = L[i,j] * R[i,jr]
    end
    verbose && print("random initial bcvumps $(Ni)×$(Nj) environment-> ")
    AL,C,AR,FL,FR
end

function _initializect_square(M::AbstractArray{<:AbstractArray, 2}, chkp_file::String, D::Int; verbose = false)
    env = load(chkp_file)["env"]
    Ni, Nj = size(M)
    verbose && print("bcvumps $(Ni)×$(Nj) environment load from $(chkp_file) -> ")   
    AL,C,AR,FL,FR = env.AL,env.C,env.AR,env.FL,env.FR
end

function bcvumps(rt::BCVUMPSRuntime; tol::Real, maxiter::Int, verbose = false)
    # initialize
    olderror = Inf

    stopfun = StopFunction(olderror, -1, tol, maxiter)
    rt, err = fixedpoint(res->bcvumpstep(res...), (rt, olderror, tol), stopfun)
    verbose && println("bcvumps done@step: $(stopfun.counter), error=$(err)")
    return rt
end

function bcvumpstep(rt::BCVUMPSRuntime,err,tol)
    M,AL,C,AR,FL,FR= rt.M,rt.AL,rt.C,rt.AR,rt.FL,rt.FR
    AL, C, AR = ACCtoALAR(AL, C, AR, M, FL, FR; tol = tol/10)
    _, FL = leftenv!(AL, M, FL; tol = tol/10)
    _, FR = rightenv!(AR, M, FR; tol = tol/10)
    err = error(AL,C,FL,M,FR)
    return SquareBCVUMPSRuntime(M, AL, C, AR, FL, FR), err, tol
end


