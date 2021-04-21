using LinearAlgebra
using KrylovKit
using Random

safesign(x::Number) = iszero(x) ? one(x) : sign(x)
"""
    qrpos(A)

Returns a QR decomposition, i.e. an isometric `Q` and upper triangular `R` matrix, where `R`
is guaranteed to have positive diagonal elements.
"""
qrpos(A) = qrpos!(copy(A))
function qrpos!(A)
    F = qr!(A)
    Q = Matrix(F.Q)
    R = F.R
    phases = safesign.(diag(R))
    rmul!(Q, Diagonal(phases))
    lmul!(Diagonal(conj!(phases)), R)
    return Q, R
end

"""
    lqpos(A)

Returns a LQ decomposition, i.e. a lower triangular `L` and isometric `Q` matrix, where `L`
is guaranteed to have positive diagonal elements.
"""
lqpos(A) = lqpos!(copy(A))
function lqpos!(A)
    F = qr!(Matrix(A'))
    Q = Matrix(Matrix(F.Q)')
    L = Matrix(F.R')
    phases = safesign.(diag(L))
    lmul!(Diagonal(phases), Q)
    rmul!(L, Diagonal(conj!(phases)))
    return L, Q
end

function cellones(Ni,Nj,D)
    Cell = Array{Array{Float64,2},2}(undef, Ni, Nj)
    for j = 1:Nj,i = 1:Ni
        Cell[i,j] = Matrix{Float64}(I, D, D)
    end
    return Cell
end

function ρmap(ρ,Ai,J)
    Nj = size(Ai,1)
    for j = 1:Nj
        jr = J+j-1 - (J+j-1 > Nj)*Nj
        ρ = ein"dc,csb,dsa -> ab"(ρ,Ai[jr],conj(Ai[jr]))
        # @tensor ρ[a,b] := ρ[a',b']*Ai[jr][b',s,b]*conj(Ai[jr][a',s,a])
    end
    return ρ
end

"""
    getL!(A,L; kwargs...)

````
┌ A1─A2─    ┌      L ─
ρ │  │    = ρ   =  │
┕ A1─A2─    ┕      L'─
````

ρ=L'*L, return L, where `L`is guaranteed to have positive diagonal elements.
L = cholesky!(ρ).U
If ρ is not exactly positive definite, cholesky will fail
"""
function getL!(A,L; kwargs...)
    Ni,Nj = size(A)
    D = size(A[1,1],1)
    for j = 1:Nj, i = 1:Ni
        _,ρs,_ = eigsolve(ρ->ρmap(ρ,A[i,:],j), L[i,j]'*L[i,j], 1, :LM; ishermitian = false, maxiter = 1, kwargs...)
        ρ = real(ρs[1] + ρs[1]')
        ρ ./= tr(ρ)
        F = svd!(ρ)
        Lo = lmul!(Diagonal(sqrt.(F.S)), F.Vt)
        _, L[i,j] = qrpos!(Lo)
    end
    return L
end

"""
    getAL(A,L)

Given an MPS tensor `A` and `L` ，return a left-canonical MPS tensor `AL`, a gauge transform `R` and
a scalar factor `λ` such that ``λ AR R = L A``
"""
function getAL(A,L)
    Ni,Nj = size(A)
    AL = Array{Array{Float64,3},2}(undef, Ni, Nj)
    Le = Array{Array{Float64,2},2}(undef, Ni, Nj)
    λ = zeros(Ni,Nj)
    for j = 1:Nj,i = 1:Ni
        D, d, = size(A[i,j])
        Q, R = qrpos!(reshape(L[i,j]*reshape(A[i,j], D, d*D), D*d, D))
        AL[i,j] = reshape(Q, D, d, D)
        λ[i,j] = norm(R)
        Le[i,j] = rmul!(R, 1/λ[i,j])
    end
    return AL, Le, λ
end

function getLsped(Le, A, AL; kwargs...)
    Ni,Nj = size(A)
    L = Array{Array{Float64,2},2}(undef, Ni, Nj)
    for j = 1:Nj,i = 1:Ni
        _, Ls, _ = eigsolve(X -> ein"dc,csb,dsa -> ab"(X,A[i,j],conj(AL[i,j])), Le[i,j], 1, :LM; ishermitian = false, kwargs...)
        _, L[i,j] = qrpos!(real(Ls[1]))
    end
    return L
end

"""
    leftorth(A,L=cellones(size(A,1),size(A,2),size(A[1,1],1)); tol = 1e-12, maxiter = 100, kwargs...)

Given an MPS tensor `A`, return a left-canonical MPS tensor `AL`, a gauge transform `L` and
a scalar factor `λ` such that ``λ AL L = L A``, where an initial guess for `L` can be
provided.
"""
function leftorth(A,L=cellones(size(A,1),size(A,2),size(A[1,1],1)); tol = 1e-12, maxiter = 100, kwargs...)
    L = getL!(A,L; kwargs...)
    AL, Le, λ= getAL(A,L;kwargs...)
    numiter = 1
    while norm(L.-Le) > tol && numiter < maxiter
        L = getLsped(Le, A, AL; kwargs...)
        AL, Le, λ = getAL(A, L; kwargs...)
        numiter += 1
    end
    L = Le
    return AL, L, λ
end


"""
    rightorth(A,L=cellones(size(A,1),size(A,2),size(A[1,1],1)); tol = 1e-12, maxiter = 100, kwargs...)

Given an MPS tensor `A`, return a gauge transform R, a right-canonical MPS tensor `AR`, and
a scalar factor `λ` such that ``λ R AR^s = A^s R``, where an initial guess for `R` can be
provided.
"""
function rightorth(A,L=cellones(size(A,1),size(A,2),size(A[1,1],1)); tol = 1e-12, maxiter = 100, kwargs...)
    Ni,Nj = size(A)
    Ar = Array{Array{Float64,3},2}(undef, Ni, Nj)
    Lr = Array{Array{Float64,2},2}(undef, Ni, Nj)
    for j = 1:Nj,i = 1:Ni
        Ar[i,j] = permutedims(A[i,j],(3,2,1))
        Lr[i,j] = permutedims(L[i,j],(2,1))
    end
    AL, L, λ = leftorth(Ar,Lr; tol = tol, kwargs...)
    R = Array{Array{Float64,2},2}(undef, Ni, Nj)
    AR = Array{Array{Float64,3},2}(undef, Ni, Nj)
    for j = 1:Nj,i = 1:Ni
        R[i,j] = permutedims(L[i,j],(2,1))
        AR[i,j] = permutedims(AL[i,j],(3,2,1))
    end
    return R, AR, λ
end

"""
    FLmap(ALi, ALip, Mi, FL, J)

ALip means ALᵢ₊₁
```
 ┌──  ALᵢⱼ  ── ALᵢⱼ₊₁   ──   ...   
 │     │        │          
FLᵢⱼ ─ Mᵢⱼ  ── Mᵢⱼ₊₁    ──   ...
 │     │        │       
 ┕──  ALᵢ₊₁ⱼ ─ ALᵢ₊₁ⱼ₊₁ ──   ...
```
"""
function FLmap(ALi, ALip, Mi, FL, J)
    Nj = size(ALi,1)
    for j=1:Nj
        jr = J+j-1 - (J+j-1 > Nj)*Nj
        FL = ein"abc,cde,bfhd,afg -> ghe"(FL,ALi[jr],Mi[jr],conj(ALip[jr]))
    end
    return FL
end

"""
    FRmap(ARi, ARip, Mi, FR, J)

ARip means ARᵢ₊₁
```
   ... ─── ARᵢⱼ₋₁  ── ARᵢⱼ  ──┐ 
            │          │      │ 
   ... ──── Mᵢⱼ₋₁  ── Mᵢⱼ  ──FRᵢⱼ
            │          │      │  
   ... ─ ARᵢ₊₁ⱼ₋₁ ─ ARᵢ₊₁ⱼ  ──┘ 
```
"""
function FRmap(ARi, ARip, Mi, FR, J)
    Nj = size(ARi,1)
    for j=1:Nj
        jr = J-(j-1) + (J-(j-1) < 1)*Nj
        FR = ein"abc,eda,hfbd,gfc -> ehg"(FR,ARi[jr],Mi[jr],conj(ARip[jr]))
    end
    return FR
end

function FLint(AL, M)
    Ni,Nj = size(AL)
    FL = Array{Array{Float64,3},2}(undef, Ni, Nj)
    for j = 1:Nj,i = 1:Ni
        D = size(AL[i,j],1)
        dL = size(M[i,j],1)
        FL[i,j] = rand(Float64, D, dL, D)
    end
    return FL
end

function FRint(AR, M)
    Ni,Nj = size(AR)
    FR = Array{Array{Float64,3},2}(undef, Ni, Nj)
    for j = 1:Nj,i = 1:Ni
        D = size(AR[i,j],1)
        dR = size(M[i,j],3)
        FR[i,j] = rand(Float64, D, dR, D)
    end
    return FR
end

"""
    leftenv!(AL, M, FL = FLint(AL,M); kwargs...)

Compute the left environment tensor for MPS A and MPO M, by finding the left fixed point
of AL - M - conj(AL) contracted along the physical dimension.
```
 ┌──  ALᵢⱼ  ── ALᵢⱼ₊₁   ──   ...         ┌── 
 │     │        │                        │   
FLᵢⱼ ─ Mᵢⱼ  ── Mᵢⱼ₊₁    ──   ...  = λLᵢⱼ FLᵢⱼ 
 │     │        │                        │   
 ┕──  ALᵢ₊₁ⱼ ─ ALᵢ₊₁ⱼ₊₁ ──   ...         ┕── 
```
"""
function leftenv!(AL, M, FL = FLint(AL,M); kwargs...)
    Ni,Nj = size(AL)
    λL = zeros(Ni,Nj)
    for j = 1:Nj,i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        λLs, FL1s, _= eigsolve(X->FLmap(AL[i,:], AL[ir,:], M[i,:], X, j), FL[i,j], 1, :LM; ishermitian = false, kwargs...)
        FL[i,j] = real(FL1s[1])
        λL[i,j] = real(λLs[1])
    end
    return λL, FL
end

"""
    rightenv!(AR, M, FR = FRint(AR,M); kwargs...)

Compute the right environment tensor for MPS A and MPO M, by finding the left fixed point
of AR - M - conj(AR) contracted along the physical dimension.
```
   ... ─── ARᵢⱼ₋₁  ── ARᵢⱼ  ──┐          ──┐   
            │          │      │            │  
   ... ──── Mᵢⱼ₋₁  ── Mᵢⱼ  ──FRᵢⱼ  = λRᵢⱼ──FRᵢⱼ
            │          │      │            │  
   ... ─ ARᵢ₊₁ⱼ₋₁ ─ ARᵢ₊₁ⱼ  ──┘          ──┘  
```
"""
function rightenv!(AR, M, FR = FRint(AR,M); kwargs...)
    Ni,Nj = size(AR)
    λR = zeros(Ni,Nj)
    for j = 1:Nj,i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        λRs, FR1s, _= eigsolve(X->FRmap(AR[i,:], AR[ir,:], M[i,:], X, j), FR[i,j], 1, :LM; ishermitian = false, kwargs...)
        FR[i,j] = real(FR1s[1])
        λR[i,j] = real(λRs[1])
    end
    return λR, FR
end

"""
    ACmap(ACij, FLj, FRj, Mj, II)

```
┌─────── ACᵢⱼ ─────┐
│        │         │          
FLᵢⱼ ─── Mᵢⱼ ───── FRᵢⱼ
│        │         │   
FLᵢ₊₁ⱼ ─ Mᵢ₊₁ⱼ ──  FRᵢ₊₁ⱼ
│        │         │    
.        .         .
.        .         .
.        .         .
```
"""
function ACmap(ACij, FLj, FRj, Mj, II)
    Ni = size(FLj,1)
    for i=1:Ni
        ir = II+i-1 - (II+i-1 > Ni)*Ni
        ACij = ein"abc,cde,bhfd,efg -> ahg"(FLj[ir],ACij,Mj[ir],FRj[ir])
    end
    return ACij
end

"""
    Cmap(Cij, FLjp, FRj, II)

```
┌────Cᵢⱼ ───┐
│           │          
FLᵢⱼ₊₁ ──── FRᵢⱼ
│           │   
FLᵢ₊₁ⱼ₊₁ ── FRᵢ₊₁ⱼ
│           │        
.           .     
.           .     
.           .     
```
"""
function Cmap(Cij, FLjp, FRj, II)
    Ni = size(FLjp,1)
    for i=1:Ni
        ir = II+i-1 - (II+i-1 > Ni)*Ni
        Cij = ein"abc,cd,dbe -> ae"(FLjp[ir],Cij,FRj[ir])
    end
    return Cij
end

"""
    ACenv(AC, FL, M, FR;kwargs...)

Compute the up environment tensor for MPS `FL`,`FR` and MPO `M`, by finding the up fixed point
        of `FL - M - FR` contracted along the physical dimension.
```
┌─────── ACᵢⱼ ─────┐
│        │         │          
FLᵢⱼ ─── Mᵢⱼ ───── FRᵢⱼ
│        │         │   
FLᵢ₊₁ⱼ ─ Mᵢ₊₁ⱼ ──  FRᵢ₊₁ⱼ  =  λACᵢⱼ ┌──── ACᵢⱼ ───┐
│        │         │                │      │      │  
.        .         .
.        .         .
.        .         .
```
"""
function ACenv!(AC, FL, M, FR; kwargs...)
    Ni,Nj = size(AC)
    λAC = zeros(Ni,Nj)
    for j = 1:Nj,i = 1:Ni
        λACs, ACs, = eigsolve(X->ACmap(X, FL[:,j], FR[:,j], M[:,j], i), AC[i,j], 1, :LM; ishermitian = false, kwargs...)
        AC[i,j] = real(ACs[1])
        λAC[i,j] = real(λACs[1])
    end
    return λAC, AC
end

"""
    Cenv(C, FL, FR;kwargs...)

Compute the up environment tensor for MPS `FL` and `FR`, by finding the up fixed point
    of `FL - FR` contracted along the physical dimension.
```
┌────Cᵢⱼ ───┐
│           │          
FLᵢⱼ₊₁ ──── FRᵢⱼ
│           │   
FLᵢ₊₁ⱼ₊₁ ── FRᵢ₊₁ⱼ   =  λCᵢⱼ ┌──Cᵢⱼ ─┐
│           │                │       │  
.           .     
.           .     
.           .     
```
"""
function Cenv!(C, FL, FR; kwargs...)
    Ni,Nj = size(C)
    λC = zeros(Ni,Nj)
    for j = 1:Nj,i = 1:Ni
        jr = j + 1 - (j==Nj) * Nj
        λCs, Cs, = eigsolve(X->Cmap(X, FL[:,jr], FR[:,j], i), C[i,j], 1, :LM; ishermitian = false, kwargs...)
        C[i,j] = real(Cs[1])
        λC[i,j] = real(λCs[1])
    end
    return λC, C
end

function ACCtoAL(ACij,Cij)
    D,d, = size(ACij)
    QAC, _ = qrpos(reshape(ACij,(D*d, D)))
    QC, _ = qrpos(Cij)
    ALij = reshape(QAC*QC', (D, d, D))
end

function ACCtoAR(ACij,Cijr)
    D,d, = size(ACij)
    _, QAC = lqpos(reshape(ACij,(D, d*D)))
    _, QC = lqpos(Cijr)
    ARij = reshape(QC'*QAC, (D, d, D))
end

"""
    itoir(i,Ni,Nj)

````
i -> (i,j) -> (i,jr) -> ir
````
"""
function itoir(i,Ni,Nj)
    Liner = LinearIndices((1:Ni,1:Nj))
    Cart = CartesianIndices((1:Ni,1:Nj))
    Index = Cart[i]
    i,j = Index[1],Index[2]
    jr = j - 1 + (j==1)*Nj
    Liner[i,jr]
end

function ALCtoAC(AL,C)
    Ni,Nj = size(AL)
    ACij = [ein"asc,cb -> asb"(AL[i],C[i]) for i=1:Ni*Nj]
    reshape(ACij,Ni,Nj)
end

"""
    ACCtoALAR(AL, C, AR, M, FL, FR; kwargs...)

QR factorization to get `AL` and `AR` from `AC` and `C`

````
──ALᵢⱼ──Cᵢⱼ──  =  ──ACᵢⱼ──  = ──Cᵢ₋₁ⱼ ──ARᵢⱼ──
  │                  │                  │   
````
"""
function ACCtoALAR(AL, C, AR, M, FL, FR; kwargs...)
    Ni,Nj = size(AL)
    AC = ALCtoAC(AL,C)
    _, AC = ACenv!(AC, FL, M, FR; kwargs...)
    _, C = Cenv!(C, FL, FR; kwargs...)

    ALij = [ACCtoAL(AC[i],C[i]) for i=1:Ni*Nj]
    AL = reshape(ALij,Ni,Nj)
    ARij = [ACCtoAR(AC[i],C[itoir(i,Ni,Nj)]) for i=1:Ni*Nj]
    AR = reshape(ARij,Ni,Nj)
    return AL, C, AR
end

"""
Compute the error through all environment `AL,C,FL,M,FR`

````
        ┌── AC──┐         
        │   │   │           ┌── AC──┐ 
MAC1 =  FL─ M ──FR  =  λAC  │   │   │ 
        │   │   │         

        ┌── AC──┐         
        │   │   │           ┌──C──┐ 
MAC2 =  FL─ M ──FR  =  λAC  │     │ 
        │   │   │         
        ┕── AL─     
        
── MAC1 ──    ≈    ── AL ── MAC2 ── 
    │                 │
````
"""
function error(AL,C,FL,M,FR)
    Ni,Nj = size(AL)
    AC = Array{Array{Float64,3},2}(undef, Ni,Nj)
    err = 0
    for j = 1:Nj,i = 1:Ni
        AC[i,j] = ein"asc,cb -> asb"(AL[i,j],C[i,j])
        MAC = ACmap(AC[i,j], FL[:,j], FR[:,j], M[:,j], i)
        MAC -= ein"asd,cpd,cpb -> asb"(AL[i,j],conj(AL[i,j]),MAC)
        err += norm(MAC)
    end
    return err
end