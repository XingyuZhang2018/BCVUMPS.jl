"""
    bcvumps_env(model::MT, β, D; tol=1e-10, maxiter=20, verbose = false) where {MT <: HamiltonianModel}

return the bcvumps environment of the `model` as a function of the inverse
temperature `β` and the environment bonddimension `D` as calculated with
vumps. Save `env` in file `./data/model_β_D.jld2`. Requires that `model_tensor` are defined for `model`.
"""
function bcvumps_env(model::MT, β, D; tol=1e-10, maxiter=20, verbose = false, atype = Array) where {MT <: HamiltonianModel}
    M = model_tensor(model, β; atype = atype)
    mkpath("./data/$(model)_$(atype)")
    chkp_file = "./data/$(model)_$(atype)/$(model)_$(atype)_β$(β)_D$(D).jld2"
    if isfile(chkp_file)                               
        rt = SquareBCVUMPSRuntime(M, chkp_file, D; verbose = verbose)   
    else
        rt = SquareBCVUMPSRuntime(M, Val(:random), D; verbose = verbose)
    end
    env = bcvumps(rt; tol=tol, maxiter=maxiter, miniter = 1, verbose = verbose)
    return env
end

"""
    Z(env::SquareBCVUMPSRuntime)

return the partition function of the `env`.
"""
function Z(env::SquareBCVUMPSRuntime)
    M,AL,C,FL,FR = env.M,env.AL,env.C,env.FL,env.FR
    Ni,Nj = size(M)
    ACij = [ein"asc,cb -> asb"(AL[i],C[i]) for i=1:Ni*Nj]
    AC = reshape(ACij,Ni,Nj)
    z_tol = 1
    for j = 1:Nj,i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        jr = j + 1 - Nj * (j==Nj)
        z = ein"(((αcβ,βsη),cpds),αpγ),ηdγ -> "(FL[i,j],AC[i,j],M[i,j],conj(AC[ir,j]),FR[i,j])
        λ = ein"((αcβ,βη),ηcγ),αγ -> "(FL[i,jr],C[i,j],FR[i,j],conj(C[ir,j]))
        z_tol *= Array(z)[]/Array(λ)[]
    end
    return z_tol^(1/Ni/Nj)
end

"""
    magnetisation(env::SquareBCVUMPSRuntime, model::MT, β)

return the magnetisation of the `model`. Requires that `mag_tensor` are defined for `model`.
"""
function magnetisation(env::SquareBCVUMPSRuntime, model::MT, β) where {MT <: HamiltonianModel}
    M,AL,C,FL,FR = env.M,env.AL,env.C,env.FL,env.FR
    Ni,Nj = size(M)
    ACij = [ein"asc,cb -> asb"(AL[i],C[i]) for i=1:Ni*Nj]
    AC = reshape(ACij,Ni,Nj)
    Mag = mag_tensor(model, β; atype = _arraytype(M[1,1]))
    mag_tol = 0
    for j = 1:Nj,i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        mag = ein"(((αcβ,βsη),cpds),αpγ),ηdγ -> "(FL[i,j],AC[i,j],Mag[i,j],conj(AC[ir,j]),FR[i,j])
        λ = ein"(((αcβ,βsη),cpds),αpγ),ηdγ -> "(FL[i,j],AC[i,j],M[i,j],conj(AC[ir,j]),FR[i,j])
        mag_tol += Array(mag)[]/Array(λ)[]
    end
    return abs(mag_tol)/Ni/Nj
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
    ACij = [ein"asc,cb -> asb"(AL[i],C[i]) for i=1:Ni*Nj]
    AC = reshape(ACij,Ni,Nj)
    Ene = energy_tensor(model, β; atype = _arraytype(M[1,1]))
    ene_tol = 0
    for j = 1:Nj,i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        ene = ein"(((αcβ,βsη),cpds),αpγ),ηdγ -> "(FL[i,j],AC[i,j],Ene[i,j],conj(AC[ir,j]),FR[i,j])
        λ = ein"(((αcβ,βsη),cpds),αpγ),ηdγ -> "(FL[i,j],AC[i,j],M[i,j],conj(AC[ir,j]),FR[i,j])
        ene_tol += Array(ene)[]/Array(λ)[]
    end
    return ene_tol/Ni/Nj
end

"""
    Z(env)

return the partition function of the observable `env`.
"""
function Z(env)
    M, ALu, Cu, _, ALd, Cd, _, FL, FR = env
    Ni,Nj = size(M)
    ACu = reshape([ein"asc,cb -> asb"(ALu[i],Cu[i]) for i=1:Ni*Nj],Ni,Nj)
    ACd = reshape([ein"asc,cb -> asb"(ALd[i],Cd[i]) for i=1:Ni*Nj],Ni,Nj)
    z_tol = 1
    for j = 1:Nj,i = 1:Ni
        ir = Ni + 1 - i
        jr = j + 1 - Nj * (j==Nj)
        z = ein"(((αcβ,βsη),cpds),αpγ),ηdγ -> "(FL[i,j],ACu[i,j],M[i,j],ACd[ir,j],FR[i,j])
        λ = ein"((αcβ,βη),ηcγ),αγ -> "(FL[i,jr],Cu[i,j],FR[i,j],Cd[ir,j])
        z_tol *= Array(z)[]/Array(λ)[]
    end
    return abs(z_tol)^(1/Ni/Nj)
end

"""
    magnetisation(env::SquareBCVUMPSRuntime, model::MT, β)

return the magnetisation of the `model`. Requires that `mag_tensor` are defined for `model`.
"""
function magnetisation(env, model::MT, β) where {MT <: HamiltonianModel}
    M, ALu, Cu, _, ALd, Cd, _, FL, FR = env
    n = 1
    fieldnames(typeof(model))[end] == :n && (n = model.n)
    Ni,Nj = size(M)
    ACu = reshape([ein"asc,cb -> asb"(ALu[i],Cu[i]) for i=1:Ni*Nj],Ni,Nj)
    ACd = reshape([ein"asc,cb -> asb"(ALd[i],Cd[i]) for i=1:Ni*Nj],Ni,Nj)
    Mag = mag_tensor(model, β; atype = _arraytype(M[1,1]))
    mag_tol = 0
    for j = 1:Nj,i = 1:Ni
        ir = Ni + 1 - i
        mag = ein"(((αcβ,βsη),cpds),αpγ),ηdγ -> "(FL[i,j],ACu[i,j],Mag[i,j],ACd[ir,j],FR[i,j])
        λ = ein"(((αcβ,βsη),cpds),αpγ),ηdγ -> "(FL[i,j],ACu[i,j],M[i,j],ACd[ir,j],FR[i,j])
        mag_tol += Array(mag)[]/Array(λ)[]/n
    end
    return abs(mag_tol)/Ni/Nj
end

"""
    energy(env::SquareBCVUMPSRuntime, model::MT, β)

return the energy of the `model` as a function of the inverse
temperature `β` and the environment bonddimension `D` as calculated with
vumps. Requires that `model_tensor` are defined for `model`.
"""
function energy(env, model::MT, β::Real) where {MT <: HamiltonianModel}
    M, ALu, Cu, _, ALd, Cd, _, FL, FR = env
    n = 1
    fieldnames(typeof(model))[end] == :n && (n = model.n)
    Ni,Nj = size(M)
    ACu = reshape([ein"asc,cb -> asb"(ALu[i],Cu[i]) for i=1:Ni*Nj],Ni,Nj)
    ACd = reshape([ein"asc,cb -> asb"(ALd[i],Cd[i]) for i=1:Ni*Nj],Ni,Nj)
    Ene = energy_tensor(model, β; atype = _arraytype(M[1,1]))
    ene_tol = 0
    for j = 1:Nj,i = 1:Ni
        ir = Ni + 1 - i
        ene = ein"(((αcβ,βsη),cpds),αpγ),ηdγ -> "(FL[i,j],ACu[i,j],Ene[i,j],ACd[ir,j],FR[i,j])
        λ = ein"(((αcβ,βsη),cpds),αpγ),ηdγ -> "(FL[i,j],ACu[i,j],M[i,j],ACd[ir,j],FR[i,j])
        ene_tol += Array(ene)[]/Array(λ)[]/n
    end
    return ene_tol/Ni/Nj
end

"""
    BigZ(env)

return the partition function of the observable `env`.
"""
function BigZ(env)
    M, ALu, Cu, ARu, ALd, Cd, ARd, _, _ = env
    Ni,Nj = size(M)
    χ, D, _ = size(ALu[1,1])
    BgM = ein"acdf,bdeg -> abcefg"(M[1,1],M[1,2])
    BgM = ein"abcdef,efghij -> abcgdhij"(BgM,BgM)
    BgM = reshape(BgM, D^2, D^2, D^2, D^2)
    Au = ein"aeb,bfc,cd -> aefd"(ALu[1,1],ALu[1,2],Cu[1,2])
    Au = reshape(Au, χ, D^2, χ)
    Ad = ein"aeb,bfc,cd -> aefd"(ALd[1,1],ALd[1,2],Cd[1,2])
    Ad = reshape(Ad, χ, D^2, χ)
    λL, BgFL = bigleftenv(ALu, ALd, M)
    FL = reshape(BgFL[1,1], χ, D^2, χ)
    λR, BgFR = bigrightenv(ARu, ARd, M)
    println("Z = $(abs.(λL).^(1/Ni/Nj)), $(abs.(λR).^(1/Ni/Nj))")
    FR = reshape(BgFR[1,2], χ, D^2, χ)
    z = ein"(((adf,abc),bdeg),fgh),ceh -> "(FL,Au,BgM,Ad,FR)
    λ = ein"((ace,ab),bcf),ef -> "(FL,Cu[1,2],FR,Cd[1,2])
    return abs(Array(z)[]/Array(λ)[])^(1/Ni/Nj)
end

"""
    magofβ(::Ising,β)

return the analytical result for the magnetisation at inverse temperature
`β` for the 2d classical ising model.
"""
magofβ(::Ising, β) = β > isingβc ? (1-sinh(2*β)^-4)^(1/8) : 0.

"""
    magofdβ(::Ising,β)

return the analytical result for the derivative of magnetisation at inverse temperature
`β` for the 2d classical ising model.
"""
magofdβ(::Ising, β) = β > isingβc ? (coth(2*β)*csch(2*β)^4)/(1-csch(2*β)^4)^(7/8) : 0.

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

"""
    Zofβ(::Ising,β)

return some the numerical integrations of analytical result for the partition function at inverse temperature
`β` for the 2d classical ising model.
"""
function Zofβ(::Ising,β)
    if β == 0.2
        return 2.08450374046259
    elseif β == 0.4
        return 2.4093664345022363
    elseif β == 0.6
        return 3.3539286863974582
    elseif β == 0.8
        return 4.96201030069517
    end
end