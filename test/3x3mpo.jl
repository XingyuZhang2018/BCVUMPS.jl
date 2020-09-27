using LinearAlgebra, TensorOperations

function me_row(Xsq,Y,I,Jtype)
    @tensor ME[a,b1,b2,c,d2,d1] := I[a',b1',c1,d1']*Xsq[Jtype[1]][a,a']*Xsq[Jtype[2]][b1,b1']*Xsq[Jtype[6]][d1',d1]*
        Y[Jtype[7]][c1,c2]*I[c2,b2',c',d2']*Xsq[Jtype[3]][b2,b2']*Xsq[Jtype[4]][c',c]*Xsq[Jtype[5]][d2',d2]
    return ME
end

function me_col(Xsq,Y,I,Jtype)
    @tensor ME[d1,d2,s2,d3,d4,s1] := I[d1',c1,d4',s1']*Xsq[Jtype[1]][d1,d1']*Xsq[Jtype[5]][d4',d4]*
        Xsq[Jtype[6]][s1',s1]*Y[Jtype[7]][c1,c2]*I[d2',s2',d3',c2]*Xsq[Jtype[2]][d2,d2']*
        Xsq[Jtype[3]][s2',s2]*Xsq[Jtype[4]][d3',d3]
    return ME
end

# define M,ME
function statmechmpo(β, h, D; r = 2.0)
    I = zeros(D,D,D,D)
    for i = 1:D
        I[i,i,i,i] = 1
    end
    
    Xsq = Array{Array,1}(undef, 2)
    X1 = zeros(D,D)
    for j = 1:D, i = 1:D
        X1[i,j] = exp(-β*h(i,j))
    end

    X2 = zeros(D,D)
    for j = 1:D, i = 1:D
        X2[i,j] = exp(-β*r*h(i,j))
    end

    Xsq[1] = sqrt(X1)
    Xsq[2] = sqrt(X2)
#     @show X1 X2 X1sq X2sq
    M = Array{Array,2}(undef, 3, 3)
    @tensor MT1[a,b,c,d] := I[a',b',c',d']*Xsq[1][a',a]*Xsq[1][b',b]*Xsq[1][c',c]*Xsq[1][d',d]
    @tensor MT2[a,b,c,d] := I[a',b',c',d']*Xsq[1][a',a]*Xsq[2][b',b]*Xsq[1][c',c]*Xsq[1][d',d]
    @tensor MT3[a,b,c,d] := I[a',b',c',d']*Xsq[2][a',a]*Xsq[2][b',b]*Xsq[2][c',c]*Xsq[2][d',d]
    M[1,1],M[1,3],M[3,1],M[3,3] = MT1, MT1, MT1, MT1
    M[1,2] = MT2
    M[2,1] = permutedims(MT2,[1,3,2,4])
    M[2,3] = permutedims(MT2,[2,1,3,4])
    M[3,2] = permutedims(MT2,[1,4,3,2])
    M[2,2] = MT3

    # For computing energy: M2 is a tensor across 2 nearest neighbor sites in the lattice, whose
    # expectation value in the converged fixed point of the transfer matrix represents the energy
    Y = Array{Array,1}(undef, 2)
    Y[1] = zeros(D,D)
    for j = 1:D, i = 1:D
        Y[1][i,j] = h(i,j)*exp(-β*h(i,j))
    end
    Y[2] = zeros(D,D)
    for j = 1:D, i = 1:D
        Y[2][i,j] = r*h(i,j)*exp(-β*r*h(i,j))
    end
    
    ME_row = Array{Array,2}(undef, 3, 3)
    ME_row[1,1] = me_row(Xsq,Y,I,[1,1,2,1,1,1,1])
    ME_row[1,2] = me_row(Xsq,Y,I,[1,2,1,1,1,1,1])
    ME_row[1,3] = me_row(Xsq,Y,I,[1,1,1,1,1,1,1])
    ME_row[2,1] = me_row(Xsq,Y,I,[1,1,2,2,2,1,2])
    ME_row[2,2] = me_row(Xsq,Y,I,[2,2,1,1,1,2,2])
    ME_row[2,3] = me_row(Xsq,Y,I,[2,1,1,2,1,1,1])
    ME_row[3,1] = me_row(Xsq,Y,I,[1,1,1,1,2,1,1])
    ME_row[3,2] = me_row(Xsq,Y,I,[1,1,1,1,1,2,1])
    ME_row[3,3] = me_row(Xsq,Y,I,[1,1,1,1,1,1,1])
    
    ME_col = Array{Array,2}(undef, 3, 3)
    ME_col[1,1] = me_col(Xsq,Y,I,[1,1,1,2,1,1,1])
    ME_col[1,2] = me_col(Xsq,Y,I,[1,2,2,2,1,1,2])
    ME_col[1,3] = me_col(Xsq,Y,I,[1,2,1,1,1,1,1])
    ME_col[2,1] = me_col(Xsq,Y,I,[1,1,1,1,2,1,1])
    ME_col[2,2] = me_col(Xsq,Y,I,[2,1,1,1,2,2,2])
    ME_col[2,3] = me_col(Xsq,Y,I,[2,1,1,1,1,1,1])
    ME_col[3,1] = me_col(Xsq,Y,I,[1,1,1,1,1,1,1])
    ME_col[3,2] = me_col(Xsq,Y,I,[1,1,2,1,1,2,1])
    ME_col[3,3] = me_col(Xsq,Y,I,[1,1,1,1,1,1,1])
    
    λM = norm(M)
    λME_row = norm(ME_row)
    λME_col = norm(ME_col)
    return M/λM, ME_row/λME_row, ME_col/λME_col, λM,λME_row,λME_col
end


classicalisingmpo(β; J = 1.0, h = 0.,r = 1.0) = statmechmpo(β, (s1,s2)->-J*(-1)^(s1!=s2) - h/2*(s1==1 + s2==1),2;r)

# function to get energy
function energy_row(M, ME, AL, C, AR, FL3, FR3,λM,λME,i,j)
    Ni,Nj = size(M)
    ir = i + 1 - (i==Ni)*Ni
    jr = j + 1 - (j==Nj)*Nj
    @tensor AAC1[α,s1,s2,β] := AL[i,j][α,s1,α']*C[i,j][α',β']*AR[i,jr][β',s2,β]
    @tensor AAC2[α,s1,s2,β] := AL[ir,j][α,s1,α']*C[ir,j][α',β']*AR[ir,jr][β',s2,β]
    @tensor Z2 = scalar(FL3[i,j][α,c,β]*AAC1[β,s1,s2,β']*M[i,j][c,t1,d,s1]*
        M[i,jr][d,t2,c',s2]*FR3[i,jr][β',c',α']*conj(AAC2[α,t1,t2,α']))
    @tensor e = scalar(FL3[i,j][α,c,β]*AAC1[β,s1,s2,β']*ME[c,t1,t2,c',s2,s1]*
        FR3[i,jr][β',c',α']*conj(AAC2[α,t1,t2,α']) / Z2)
    return e/λM^2*λME
end

function energy_col(M, ME, AL, C, AR, FL4, FR4,λM,λME, i, j)
    Ni,Nj = size(M)
    jr = j + 1 - (j==Nj)*Nj
    ir = i + 1 - (i==Ni)*Ni
    irr = i + 2 - (i+2>Ni)*Ni - (Ni==1)
    @tensor AC1[β,s,β'] := AL[i,j][β,s,d1]*C[i,j][d1,β']
    @tensor AC2[β,s,β'] := AL[irr,j][β,s,d1]*C[irr,j][d1,β']
    @tensor Z2 = scalar(FL4[i,j][α,a,b,β]*AC1[β,s,β']*M[i,j][b,c,b',s]*
        M[ir,j][a,s',a',c]*FR4[i,j][β',b',a',α']*conj(AC2[α,s',α']))
    @tensor e = scalar(FL4[i,j][α,a,b,β]*AC1[β,s,β']*ME[b,a,s',a',b',s]*
        FR4[i,j][β',b',a',α']*conj(AC2[α,s',α']) / Z2)
    return e/λM^2*λME
end

function energy(M, ME_row, ME_col, AL, C, AR, FL3,FL4, FR3, FR4,λM,λME_row, λME_col)
    Ni, Nj = size(M)
    e_row = 0
    e_col = 0
    for j = 1:Nj, i = 1:Ni
        row = energy_row(M, ME_row[i,j], AL, C, AR, FL3, FR3,λM,λME_row,i,j)
        e_row += row
        col = energy_col(M, ME_col[i,j], AL, C, AR, FL4, FR4,λM,λME_col,i,j)
        e_col += col
#         @show i j row col
    end
#     @show e_row e_col
    return (e_row + e_col)/Ni/Nj
end
