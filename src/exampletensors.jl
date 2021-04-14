using JLD2
using FileIO

# tensor for classical 2-d model
const isingβc = log(1+sqrt(2))/2

"""
    model_tensor(model::Ising, β::Real)

return the  `MT <: HamiltonianModel` bulktensor at inverse temperature `β` for  two-dimensional
square lattice tensor-network.
"""
function model_tensor(model::Ising, β::Real)
    Ni,Nj = model.Ni, model.Nj
    ham = [-1. 1;1 -1]
    w = exp.(- β .* ham)
    wsq = sqrt(w)
    m = ein"ia,ib,ic,id -> abcd"(wsq,wsq,wsq,wsq)
    M = Array{Array,2}(undef, Ni, Nj)

    for j = 1:Nj, i = 1:Ni
        M[i,j] = m
    end
    return M
end

function model_tensor(model::Ising22, β::Real)
    r = model.r
    wsq, w = Array{Array,1}(undef, 2), Array{Array,1}(undef, 2)
    ham = [-1. 1;1 -1]
    w[1] = exp.(- β .* ham)
    w[2] = exp.(- r * β .* ham)
    wsq = sqrt.(w)
    M = Array{Array,2}(undef, 2, 2)
    MT = ein"ia,ib,ic,id -> abcd"(wsq[1],wsq[2],wsq[2],wsq[1])
    M[1,1] = MT
    M[1,2] = permutedims(MT,[3,2,1,4])
    M[2,1] = permutedims(MT,[1,4,3,2])
    M[2,2] = permutedims(MT,[3,4,1,2])
    return M
end

function model_tensor(model::Ising33, β::Real)
    r = model.r
    wsq, w = Array{Array,1}(undef, 2), Array{Array,1}(undef, 2)
    ham = [-1. 1;1 -1]
    w[1] = exp.(- β .* ham)
    w[2] = exp.(- r * β .* ham)
    wsq = sqrt.(w)
    M = Array{Array,2}(undef, 3, 3)
    MT = ein"ia,ib,ic,id -> abcd"(wsq[1],wsq[2],wsq[2],wsq[1])
    M[1,1] = MT
    M[1,2] = permutedims(MT,[3,2,1,4])
    M[2,1] = permutedims(MT,[1,4,3,2])
    M[2,2] = permutedims(MT,[3,4,1,2])
    return M
end

"""
    mag_tensor(::MT, β::Real)

return the  `MT <: HamiltonianModel` the operator for the magnetisation at inverse temperature `β` for a two-dimensional
square lattice tensor-network. 
"""
function mag_tensor(model::Ising, β::Real)
    Ni,Nj = model.Ni, model.Nj
    a = reshape(Float64[1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 -1] , 2,2,2,2)
    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))
    q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    m = ein"abcd,ai,bj,ck,dl -> ijkl"(a,q,q,q,q)
    M = Array{Array,2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        M[i,j] = m
    end
    return M
end

function mag_tensor(model::Ising22, β::Real)
    r = model.r
    a = reshape(Float64[1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 -1] , 2,2,2,2)
    wsq, w = Array{Array,1}(undef, 2), Array{Array,1}(undef, 2)
    ham = [-1. 1;1 -1]
    w[1] = exp.(- β .* ham)
    w[2] = exp.(- r * β .* ham)
    wsq = sqrt.(w)
    M = Array{Array,2}(undef, 2, 2)
    MT = ein"ijkl,ia,jb,kc,ld -> abcd"(a,wsq[1],wsq[2],wsq[2],wsq[1])
    M[1,1] = MT
    M[1,2] = permutedims(MT,[3,2,1,4])
    M[2,1] = permutedims(MT,[1,4,3,2])
    M[2,2] = permutedims(MT,[3,4,1,2])
    return M
end

"""
    energy_tensor(model::Ising, β::Real)

return the  `MT <: HamiltonianModel` the operator for the energy at inverse temperature `β` for a two-dimensional
    square lattice tensor-network. 
"""
function energy_tensor(model::Ising, β::Real)
    Ni,Nj = model.Ni, model.Nj
    ham = [-1 1;1 -1]
    w = exp.(-β .* ham)
    we = ham .* w
    wsq = sqrt(w)
    wsqi = wsq^(-1)
    e = (ein"ai,im,bm,cm,dm -> abcd"(wsqi,we,wsq,wsq,wsq) + ein"am,bi,im,cm,dm -> abcd"(wsq,wsqi,we,wsq,wsq) + 
        ein"am,bm,ci,im,dm -> abcd"(wsq,wsq,wsqi,we,wsq) + ein"am,bm,cm,di,im -> abcd"(wsq,wsq,wsq,wsqi,we)) ./ 2
    E = Array{Array,2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        E[i,j] = e
    end
    return E
end

function energy_tensor(model::Ising22, β::Real)
    r = model.r
    wsq, w, we = Array{Array,1}(undef, 2), Array{Array,1}(undef, 2), Array{Array,1}(undef, 2)
    ham = [-1. 1;1 -1]
    w[1] = exp.(- β .* ham)
    w[2] = exp.(- r * β .* ham)
    we[1] = ham .* w[1]
    we[2] = r .* ham .* w[2]
    wsq = sqrt.(w)
    wsqi = wsq.^(-1)
    
    M = Array{Array,2}(undef, 2, 2)
    MT = (ein"ai,im,bm,cm,dm -> abcd"(wsqi[1],we[1],wsq[2],wsq[2],wsq[1]) + 
              ein"am,bi,im,cm,dm -> abcd"(wsq[1],wsqi[2],we[2],wsq[2],wsq[1]) + 
              ein"am,bm,ci,im,dm -> abcd"(wsq[1],wsq[2],wsqi[2],we[2],wsq[1]) + 
              ein"am,bm,cm,di,im -> abcd"(wsq[1],wsq[2],wsq[2],wsqi[1],we[1])) ./ 2
    M[1,1] = MT
    M[1,2] = permutedims(MT,[3,2,1,4])
    M[2,1] = permutedims(MT,[1,4,3,2])
    M[2,2] = permutedims(MT,[3,4,1,2])
    return M
end

