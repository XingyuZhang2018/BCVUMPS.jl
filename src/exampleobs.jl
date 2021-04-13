"""
    obs_env()

If `Ni,Nj>2` and `Mij` are different bulk tensor, the up and down environment are different. So to calculate observable, we must get ACup and ACdown, which is easy to get by overturning the `Mij`. Then be cautious to get the new `FL` and `FR` environment.
"""
function obs_env()
    
end

"""
    bcvumps_env(model::MT, Ni, Nj, β, D; tol=1e-10, maxiter=100, verbose = false) where {MT <: HamiltonianModel}

return the bcvumps environment of the `model` as a function of the inverse
temperature `β` and the environment bonddimension `D` as calculated with
vumps. Save `env` in file `./data/model_β_D.jld2`. Requires that `model_tensor` are defined for `model`.
"""
function bcvumps_env(model::MT, β, D; tol=1e-10, maxiter=20, verbose = false) where {MT <: HamiltonianModel}
    M = model_tensor(model, β)
    mkpath("./data/")
    chkp_file = "./data/$(model)_β$(β)_D$(D).jld2"
    if isfile(chkp_file)                               
        rt = SquareBCVUMPSRuntime(M, chkp_file, D; verbose = verbose)   
    else
        rt = SquareBCVUMPSRuntime(M, Val(:random), D; verbose = verbose)
    end
    env = bcvumps(rt; tol=tol, maxiter=maxiter, verbose = verbose)
    return env
end

"""
    magnetisation(env::SquareVUMPSRuntime, model::MT, β; r=1)

return the magnetisation of the `model`. Requires that `mag_tensor` are defined for `model`.
"""
function magnetisation(env::SquareBCVUMPSRuntime, model::MT, β) where {MT <: HamiltonianModel}
    M,AL,C,FL,FR = env.M,env.AL,env.C,env.FL,env.FR
    Ni,Nj = size(M)
    AC = Array{Array,2}(undef, Ni, Nj)
    for j = 1:Nj,i = 1:Ni
        AC[i,j] = ein"asc,cb -> asb"(AL[i,j],C[i,j])
    end
    Mag = mag_tensor(model, β)
    mag_tol = 0
    for j = 1:Nj,i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        mag = ein"αcβ,βsη,cpds,ηdγ,αpγ ->"(FL[i,j],AC[i,j],Mag[i,j],FR[i,j],conj(AC[ir,j]))[]
        λ = ein"αcβ,βsη,cpds,ηdγ,αpγ ->"(FL[i,j],AC[i,j],M[i,j],FR[i,j],conj(AC[ir,j]))[]
        mag_tol += abs(mag/λ)
    end
    return mag_tol/Ni/Nj
end

"""
    energy(env::SquareBCVUMPSRuntime, model::MT, β)

return the energy of the `model` as a function of the inverse
temperature `β` and the environment bonddimension `D` as calculated with
vumps. Requires that `model_tensor` are defined for `model`.
"""
function energy(env::SquareBCVUMPSRuntime, model::MT, β::Real) where {MT <: HamiltonianModel}
    M,AL,C,FL,FR = env.M,env.AL,env.C,env.FL,env.FR
    Ni,Nj = size(M)
    AC = Array{Array,2}(undef, Ni, Nj)
    for j = 1:Nj,i = 1:Ni
        AC[i,j] = ein"asc,cb -> asb"(AL[i,j],C[i,j])
    end
    Ene = energy_tensor(model, β)
    ene_tol = 0
    for j = 1:Nj,i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        ene = ein"αcβ,βsη,cpds,ηdγ,αpγ ->"(FL[i,j],AC[i,j],Ene[i,j],FR[i,j],conj(AC[ir,j]))[]
        λ = ein"αcβ,βsη,cpds,ηdγ,αpγ ->"(FL[i,j],AC[i,j],M[i,j],FR[i,j],conj(AC[ir,j]))[]
        ene_tol += ene/λ
    end
    return ene_tol/Ni/Nj
end

"""
    magofβ(::Ising,β)

return the analytical result for the magnetisation at inverse temperature
`β` for the 2d classical ising model.
"""
magofβ(::Ising, β) = β > isingβc ? (1-sinh(2*β)^-4)^(1/8) : 0.

"""
    eneofβ(::Ising,β)

return some the numerical integrations of analytical result for the energy at inverse temperature
`β` for the 2d classical ising model.
"""
function eneofβ(::Ising, β)
    if β == 0.2
        return -0.42822885693016843
    elseif β == 0.4
        return -1.1060792706185651
    elseif β == 0.6
        return -1.909085845408498
    elseif β == 0.8
        return -1.9848514445364174
    end
end