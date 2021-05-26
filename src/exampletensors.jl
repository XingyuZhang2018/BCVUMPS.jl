using JLD2
using FileIO

# tensor for classical 2-d model
const isingβc = log(1+sqrt(2))/2

"""
    model_tensor(model::Ising, β::Real)

return the  `MT <: HamiltonianModel` bulktensor at inverse temperature `β` for  two-dimensional
square lattice tensor-network.
"""
function model_tensor(model::Ising, β::Real; atype = Array)
    Ni, Nj = model.Ni, model.Nj
    ham = [-1. 1;1 -1]
    w = exp.(- β .* ham)
    wsq = sqrt(w)
    m = atype(ein"ia,ib,ic,id -> abcd"(wsq, wsq, wsq, wsq))
    reshape([m for i=1:Ni*Nj], Ni, Nj)
end

function model_tensor(model::Ising22, β::Real; atype = Array)
    r = model.r
    ham = [-1. 1;1 -1]
    w1 = exp.(- β .* ham)
    w2 = exp.(- r * β .* ham)
    wsq1 = sqrt(w1)
    wsq2 = sqrt(w2)
    M11 = atype(ein"ia,ib,ic,id -> abcd"(wsq1, wsq2, wsq2, wsq1))
    M12 = permutedims(M11, [3,2,1,4])
    M21 = permutedims(M11, [1,4,3,2])
    M22 = permutedims(M11, [3,4,1,2])
    reshape([M11,M21,M12,M22], 2, 2)
end

########## To do: correct Ising33 ##########
function model_tensor(model::Ising33, β::Real; atype = Array)
    r = model.r
    ham = [-1. 1;1 -1]
    w1 = exp.(- β .* ham)
    w2 = exp.(- r * β .* ham)
    wsq1 = sqrt(w1)
    wsq2 = sqrt(w2)
    M11 = atype(ein"ia,ib,ic,id -> abcd"(wsq1, wsq2, wsq2, wsq1))
    M12 = permutedims(M11, [3,2,1,4])
    M21 = permutedims(M11, [1,4,3,2])
    M22 = permutedims(M11, [3,4,1,2])
    reshape([M11,M21,M12,M22], 2, 2)
end

"""
    mag_tensor(::MT, β::Real)

return the  `MT <: HamiltonianModel` the operator for the magnetisation at inverse temperature `β` for a two-dimensional
square lattice tensor-network. 
"""
function mag_tensor(model::Ising, β::Real; atype = Array)
    Ni,Nj = model.Ni, model.Nj
    a = reshape(Float64[1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 -1] , 2,2,2,2)
    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))
    q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    m = atype(ein"abcd,ai,bj,ck,dl -> ijkl"(a,q,q,q,q))
    reshape([m for i=1:Ni*Nj], Ni, Nj)
end

function mag_tensor(model::Ising22, β::Real; atype = Array)
    r = model.r
    a = reshape(Float64[1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 -1] , 2,2,2,2)
    ham = [-1. 1;1 -1]
    w1 = exp.(- β .* ham)
    w2 = exp.(- r * β .* ham)
    wsq1 = sqrt(w1)
    wsq2 = sqrt(w2)
    M11 = atype(ein"ijkl,ia,jb,kc,ld -> abcd"(a,wsq1,wsq2,wsq2,wsq1))
    M12 = permutedims(M11, [3,2,1,4])
    M21 = permutedims(M11, [1,4,3,2])
    M22 = permutedims(M11, [3,4,1,2])
    reshape([M11,M21,M12,M22], 2, 2)
end

"""
    energy_tensor(model::Ising, β::Real)

return the  `MT <: HamiltonianModel` the operator for the energy at inverse temperature `β` for a two-dimensional
    square lattice tensor-network. 
"""
function energy_tensor(model::Ising, β::Real; atype = Array)
    Ni,Nj = model.Ni, model.Nj
    ham = [-1 1;1 -1]
    w = exp.(-β .* ham)
    we = ham .* w
    wsq = sqrt(w)
    wsqi = wsq^(-1)
    e = atype(ein"ai,im,bm,cm,dm -> abcd"(wsqi,we,wsq,wsq,wsq) + ein"am,bi,im,cm,dm -> abcd"(wsq,wsqi,we,wsq,wsq) + 
        ein"am,bm,ci,im,dm -> abcd"(wsq,wsq,wsqi,we,wsq) + ein"am,bm,cm,di,im -> abcd"(wsq,wsq,wsq,wsqi,we)) / 2
    reshape([e for i=1:Ni*Nj], Ni, Nj)
end

function energy_tensor(model::Ising22, β::Real; atype = Array)
    r = model.r
    ham = [-1. 1;1 -1]
    w1 = exp.(- β .* ham)
    w2 = exp.(- r * β .* ham)
    we1 = ham .* w1
    we2 = r .* ham .* w2
    wsq1 = sqrt(w1)
    wsq2 = sqrt(w2)
    wsqi1 = wsq1^(-1)
    wsqi2 = wsq2^(-1)
    
    M11 = atype(ein"ai,im,bm,cm,dm -> abcd"(wsqi1,we1,wsq2,wsq2,wsq1) + 
              ein"am,bi,im,cm,dm -> abcd"(wsq1,wsqi2,we2,wsq2,wsq1) + 
              ein"am,bm,ci,im,dm -> abcd"(wsq1,wsq2,wsqi2,we2,wsq1) + 
              ein"am,bm,cm,di,im -> abcd"(wsq1,wsq2,wsq2,wsqi1,we1)) / 2
    M12 = permutedims(M11, [3,2,1,4])
    M21 = permutedims(M11, [1,4,3,2])
    M22 = permutedims(M11, [3,4,1,2])
    reshape([M11,M21,M12,M22], 2, 2)
end

