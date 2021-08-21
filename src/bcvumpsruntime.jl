using LinearAlgebra
using KrylovKit
using Zygote

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
struct BCVUMPSRuntime{LT,T,N,AT <: AbstractArray{<:AbstractArray,2},CT,ET}
    M::AT
    AL::ET
    C::CT
    AR::ET
    FL::ET
    FR::ET
    function BCVUMPSRuntime{LT}(M::AT, AL::ET, C::CT, AR::ET, FL::ET, FR::ET) where {LT <: AbstractLattice,AT <: AbstractArray{<:AbstractArray,2}, CT <: AbstractArray{<:AbstractArray,2}, ET <: AbstractArray{<:AbstractArray,2}}
        T, N = eltype(M[1,1]), ndims(M[1,1])
        new{LT,T,N,AT,CT,ET}(M, AL, C, AR, FL, FR)
    end
end

const SquareBCVUMPSRuntime{T,AT} = BCVUMPSRuntime{SquareLattice,T,4,AT}
function SquareBCVUMPSRuntime(M::AT, AL, C, AR, FL, FR) where {AT <: AbstractArray{<:AbstractArray,2}}
    ndims(M[1,1]) == 4 || throw(DimensionMismatch("M dimensions error, should be `4`, got $(ndims(M[1,1]))."))
    BCVUMPSRuntime{SquareLattice}(M, AL, C, AR, FL, FR)
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

julia> M = Array{Array{Float64,3},2}(undef, Ni, Nj);

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
function SquareBCVUMPSRuntime(M::AbstractArray{<:AbstractArray,2}, env, D::Int; verbose=false)
    return SquareBCVUMPSRuntime(M, _initializect_square(M, env, D; verbose=verbose)...)
end

function _initializect_square(M::AbstractArray{<:AbstractArray,2}, env::Val{:random}, D::Int; verbose=false)
    A = initialA(M, D)
    AL, L = leftorth(A)
    R, AR = rightorth(AL)
    _, FL = leftenv(AL, AL, M)
    _, FR = rightenv(AR, AR, M)
    C = LRtoC(L,R)
    Ni, Nj = size(M)
    verbose && print("random initial $(Ni)×$(Nj) bcvumps_χ$(D) environment-> ")
    AL, C, AR, FL, FR
end

function _initializect_square(M::AbstractArray{<:AbstractArray,2}, chkp_file::String, D::Int; verbose=false)
    env = load(chkp_file)["env"]
    Ni, Nj = size(M)
    atype = _arraytype(M[1,1])
    verbose && print("bcvumps $(Ni)×$(Nj) environment load from $(chkp_file) -> ")   
    AL, C, AR, FL, FR = env.AL, env.C, env.AR, env.FL, env.FR
    Zygote.@ignore begin
        AL, C, AR, FL, FR = Array{atype{Float64,3},2}(env.AL), Array{atype{Float64,2},2}(env.C), Array{atype{Float64,3},2}(env.AR), Array{atype{Float64,3},2}(env.FL), Array{atype{Float64,3},2}(env.FR)
    end
    AL, C, AR, FL, FR
end

function bcvumps(rt::BCVUMPSRuntime; tol::Real, maxiter::Int, miniter::Int, verbose=false)
    # initialize
    olderror = Inf

    stopfun = StopFunction(olderror, -1, tol, maxiter, miniter)
    rt, err = fixedpoint(res -> bcvumpstep(res...), (rt, olderror), stopfun)
    verbose && println("bcvumps done@step: $(stopfun.counter), error=$(err)")
    return rt
end

function bcvumpstep(rt::BCVUMPSRuntime, err)
    M, AL, C, AR, FL, FR = rt.M, rt.AL, rt.C, rt.AR, rt.FL, rt.FR
    AC = ALCtoAC(AL,C)
    # Ni,Nj = size(AC)
    # ACp = reshape([ACmap(AC[k], FL[:,ktoij(k,Ni,Nj)[2]], FR[:,ktoij(k,Ni,Nj)[2]], M[:,ktoij(k,Ni,Nj)[2]], ktoij(k,Ni,Nj)[1]) for k=1:Ni*Nj],(Ni,Nj))
    # Cp = reshape([Cmap(C[k], FL[:,ktoij(k,Ni,Nj)[2] + 1 - (ktoij(k,Ni,Nj)[2]==Nj) * Nj], FR[:,ktoij(k,Ni,Nj)[2]], ktoij(k,Ni,Nj)[1]) for k=1:Ni*Nj],(Ni,Nj))
    # ACp = reshape([ACp[k] / sqrt(ein"abc, abc ->"(ACp[k], ACp[k])[]) for k=1:Ni*Nj],(Ni,Nj))
    # Cp = reshape([Cp[k] / sqrt(ein"ab, ab ->"(Cp[k], Cp[k])[]) for k=1:Ni*Nj],(Ni,Nj))
    _, ACp = ACenv(AC, FL, M, FR)
    _, Cp = Cenv(C, FL, FR)
    ALp, ARp = ACCtoALAR(ACp, Cp)
    _, FL = leftenv(AL, ALp, M, FL)
    _, FR = rightenv(AR, ARp, M, FR)
    _, ACp = ACenv(ACp, FL, M, FR)
    _, Cp = Cenv(Cp, FL, FR)
    ALp, ARp = ACCtoALAR(ACp, Cp)
    err = error(ALp, Cp, FL, M, FR)
    return SquareBCVUMPSRuntime(M, ALp, Cp, ARp, FL, FR), err
end

"""
    uptodown(i,Ni,Nj)

````
i -> (i,j) -> (Nj +1 - i,j) -> ir
````
"""
function uptodown(i,Ni,Nj)
    Liner = LinearIndices((1:Ni,1:Nj))
    Cart = CartesianIndices((1:Ni,1:Nj))
    Index = Cart[i]
    i,j = Index[1],Index[2]
    ir = Ni + 1 - i
    Liner[ir,j]
end

"""
    Mu, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR = obs_bcenv(model::MT, Mu::AbstractArray; atype = Array, D::Int, χ::Int, verbose = false)

If `Ni,Nj>1` and `Mij` are different bulk tensor, the up and down environment are different. So to calculate observable, we must get ACup and ACdown, which is easy to get by overturning the `Mij`. Then be cautious to get the new `FL` and `FR` environment.
"""
function obs_bcenv(model::MT, Mu::AbstractArray; atype = Array, D::Int, χ::Int, tol::Real=1e-10, maxiter::Int=10, miniter::Int=1, folder::String="./data/", verbose = false, savefile = false) where {MT <: HamiltonianModel}
    mkpath(folder*"$(model)_$(atype)")
    chkp_file_up = folder*"$(model)_$(atype)/up_D$(D)_chi$(χ).jld2"
    verbose && print("↑ ")
    if isfile(chkp_file_up)                               
        rtup = SquareBCVUMPSRuntime(Mu, chkp_file_up, χ; verbose = verbose)   
    else
        rtup = SquareBCVUMPSRuntime(Mu, Val(:random), χ; verbose = verbose)
    end
    envup = bcvumps(rtup; tol=tol, maxiter=maxiter, miniter=miniter, verbose = verbose)
    ALu,ARu,Cu = envup.AL,envup.AR,envup.C

    Zygote.@ignore savefile && begin
        ALs, Cs, ARs, FLs, FRs = Array{Array{Float64,3},2}(envup.AL), Array{Array{Float64,2},2}(envup.C), Array{Array{Float64,3},2}(envup.AR), Array{Array{Float64,3},2}(envup.FL), Array{Array{Float64,3},2}(envup.FR)
        envsave = SquareBCVUMPSRuntime(Mu, ALs, Cs, ARs, FLs, FRs)
        save(chkp_file_up, "env", envsave)
    end

    Ni, Nj = size(ALu)
    Md = [permutedims(Mu[uptodown(i,Ni,Nj)], (1,4,3,2)) for i = 1:Ni*Nj]
    Md = reshape(Md, Ni, Nj)

    chkp_file_down = folder*"$(model)_$(atype)/down_D$(D)_chi$(χ).jld2"
    verbose && print("↓ ")
    if isfile(chkp_file_down)                               
        rtdown = SquareBCVUMPSRuntime(Md, chkp_file_down, χ; verbose = verbose)   
    else
        rtdown = SquareBCVUMPSRuntime(Md, Val(:random), χ; verbose = verbose)
    end
    envdown = bcvumps(rtdown; tol=tol, maxiter=maxiter, miniter=miniter, verbose = verbose)

    Zygote.@ignore savefile && begin
        ALs, Cs, ARs, FLs, FRs = Array{Array{Float64,3},2}(envdown.AL), Array{Array{Float64,2},2}(envdown.C), Array{Array{Float64,3},2}(envdown.AR), Array{Array{Float64,3},2}(envdown.FL), Array{Array{Float64,3},2}(envdown.FR)
        envsave = SquareBCVUMPSRuntime(Md, ALs, Cs, ARs, FLs, FRs)
        save(chkp_file_down, "env", envsave)
    end
    ALd,ARd,Cd = envdown.AL,envdown.AR,envdown.C

    chkp_file_obs = folder*"$(model)_$(atype)/obs_D$(D)_chi$(χ).jld2"
    if isfile(chkp_file_obs)   
        verbose && println("←→ observable environment load from $(chkp_file_obs)")
        FL, FR = load(chkp_file_obs)["env"]
    else
        FL, FR = envup.FL,envup.FR
    end
    _, FL = obs_FL(ALu, ALd, Mu, FL)
    _, FR = obs_FR(ARu, ARd, Mu, FR)

    Zygote.@ignore savefile && begin
        envsave = (FL, FR)
        save(chkp_file_obs, "env", envsave)
    end

    Mu, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR
end


"""
    Mu, ALu, Cu, ARu = obs_bcenv_oneside(model::MT, Mu::AbstractArray; atype = Array, D::Int, χ::Int, tol::Real=1e-10, maxiter::Int=10, miniter::Int=1, folder::String="./data/", verbose = false, savefile = false) where {MT <: HamiltonianModel}

sometimes the finally observable is symetric, so we can use the same up and down environment. 
"""
function obs_bcenv_oneside(model::MT, Mu::AbstractArray; atype = Array, D::Int, χ::Int, tol::Real=1e-10, maxiter::Int=10, miniter::Int=1, folder::String="./data/", verbose = false, savefile = false) where {MT <: HamiltonianModel}
    mkpath(folder*"$(model)_$(atype)")
    chkp_file_up = folder*"$(model)_$(atype)/up_D$(D)_chi$(χ).jld2"
    verbose && print("↑ ")
    if isfile(chkp_file_up)                               
        rtup = SquareBCVUMPSRuntime(Mu, chkp_file_up, χ; verbose = verbose)   
    else
        rtup = SquareBCVUMPSRuntime(Mu, Val(:random), χ; verbose = verbose)
    end
    envup = bcvumps(rtup; tol=tol, maxiter=maxiter, miniter=miniter, verbose = verbose)
    ALu,ARu,Cu,FL,FR = envup.AL,envup.AR,envup.C,envup.FL,envup.FR

    Zygote.@ignore savefile && begin
        ALs, Cs, ARs, FLs, FRs = Array{Array{Float64,3},2}(envup.AL), Array{Array{Float64,2},2}(envup.C), Array{Array{Float64,3},2}(envup.AR), Array{Array{Float64,3},2}(envup.FL), Array{Array{Float64,3},2}(envup.FR)
        envsave = SquareBCVUMPSRuntime(Mu, ALs, Cs, ARs, FLs, FRs)
        save(chkp_file_up, "env", envsave)
    end
    Mu, ALu, Cu, ARu, FL, FR
end
